/* The following program solves the simple advection equation using the finite volume
 * method. It employs the riemann solver. First order upwind is used for discretisation
 *
 * PDE : u_t + au_x = 0
 * Boundary conditions can be set up using setupBC()*/
static char help[] = "Nonlinear, time-dependent PDE in 2d.\n";


#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscsf.h> /* For SplitFaces() */

#define DIM 2                   /* Geometric dimension */
#define ALEN(a) (sizeof(a)/sizeof((a)[0]))

typedef struct _n_FunctionalLink *FunctionalLink;
/* 'User' implements a discretization of a continuous model. */
typedef struct _n_User *User;
/* Represents continuum physical equations. */
typedef struct _n_Physics *Physics;
/* Physical model includes boundary conditions, initial conditions, and functionals of interest. It is
 * discretization-independent, but its members depend on the scenario being solved. */
typedef struct _n_Model *Model;
/* 'User' implements a discretization of a continuous model. */
typedef struct _n_User *User;

/* Declaration of all functions */
extern PetscErrorCode MyTSMonitor(TS, PetscInt, PetscReal, Vec, void *);

typedef PetscErrorCode (*SolutionFunction)(Model, PetscReal, const PetscReal *, PetscScalar *, void *);

typedef PetscErrorCode (*SetUpBCFunction)(PetscDS, Physics);

PETSC_STATIC_INLINE PetscReal Dot2Real(const PetscReal *x, const PetscReal *y) { return x[0] * y[0] + x[1] * y[1]; }

PETSC_STATIC_INLINE PetscReal Norm2Real(const PetscReal *x) { return PetscSqrtReal(PetscAbsReal(Dot2Real(x, x))); }

PETSC_STATIC_INLINE void Waxpy2Real(PetscReal a, const PetscReal *x, const PetscReal *y, PetscReal *w) {
    w[0] = a * x[0] + y[0];
    w[1] = a * x[1] + y[1];
}

/* Defining super and sub classes of user context structs */
struct _n_FunctionalLink {
    char *name;
    void *ctx;
    PetscInt offset;
    FunctionalLink next;
};

struct _n_User {
    Model model;
};

struct _n_Physics {
    PetscRiemannFunc riemann;
    PetscInt dof;          /* number of degrees of freedom per cell */
    PetscReal maxspeed;     /* kludge to pick initial time step, need to add monitoring and step control */
    void *data;
    PetscInt nfields;
    const struct FieldDescription *field_desc;
};

struct _n_Model {
    MPI_Comm comm;        /* Does not do collective communicaton, but some error conditions can be collective */
    Physics physics;
    PetscInt maxComputed;
    PetscInt numMonitored;
    FunctionalLink *functionalMonitored;
    FunctionalLink *functionalCall;
    SolutionFunction solution;
    SetUpBCFunction setupbc;
    void *solutionctx;
    PetscReal maxspeed;    /* estimate of global maximum speed (for CFL calculation) */
    PetscReal bounds[2 * DIM];
    DMBoundaryType bcs[3];
};

/* Defining the context specific for the advection model */
typedef struct {
    PetscReal center[DIM];
    PetscReal radius;
} Physics_Advect_Bump;

typedef struct {
    PetscReal inflowState;
    union {
        Physics_Advect_Bump bump;
    } sol;
    struct {
        PetscInt Solution;
        PetscInt Error;
    } functional;
} Physics_Advect;

/* Functions responsible for defining the boundary cells (elements) */
static PetscErrorCode
PhysicsBoundary_Advect_Inflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx) {
    Physics phys = (Physics) ctx;
    Physics_Advect *advect = (Physics_Advect *) phys->data;
    PetscFunctionBeginUser;
    xG[0] = advect->inflowState;
    PetscFunctionReturn(0);
}

static PetscErrorCode
PhysicsBoundary_Advect_Outflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx) {
    PetscFunctionBeginUser;
    xG[0] = xI[0];
    PetscFunctionReturn(0);
}

/* Specifying the Riemann function for the advection model */
static void
PhysicsRiemann_Advect(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, Physics phys) {
    PetscReal wind[DIM], wn;
    wind[0] = -qp[1];
    wind[1] = qp[0];
    wn = Dot2Real(wind, n);
    flux[0] = (wn > 0 ? xL[0] : xR[0]) * wn;
}

/* Routine for defining the initial solution */
static PetscErrorCode PhysicsSolution_Advect(Model mod, PetscReal time, const PetscReal *x, PetscScalar *u, void *ctx) {
    Physics phys = (Physics) ctx;
    Physics_Advect *advect = (Physics_Advect *) phys->data;
    PetscFunctionBeginUser;
    Physics_Advect_Bump *bump = &advect->sol.bump;
    PetscReal x0[DIM], v[DIM], r, cost, sint;
    cost = PetscCosReal(time);
    sint = PetscSinReal(time);
    x0[0] = cost * x[0] + sint * x[1];
    x0[1] = -sint * x[0] + cost * x[1];
    Waxpy2Real(-1, bump->center, x0, v);
    r = Norm2Real(v);
    u[0] = 0.5 + 0.5 * PetscCosReal(PetscMin(r / bump->radius, 1) * PETSC_PI);
    PetscFunctionReturn(0);
}

/* Defining the function responsible for the setting up the BCs */
static PetscErrorCode SetUpBC_Advect(PetscDS prob, Physics phys) {
    PetscErrorCode ierr;
    const PetscInt inflowids[] = {100, 200, 300}, outflowids[] = {101};
    PetscFunctionBeginUser;
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "inflow", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Advect_Inflow, ALEN(inflowids), inflowids, phys);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "outflow", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Advect_Outflow, ALEN(outflowids), outflowids, phys);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* Initializing all of the structs related to the advection model */
static PetscErrorCode PhysicsCreate_Advect(Model mod, Physics phys) {
    Physics_Advect *advect;
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    phys->riemann = (PetscRiemannFunc) PhysicsRiemann_Advect;
    ierr = PetscNew(&advect); CHKERRQ(ierr);
    phys->data = advect;
    mod->setupbc = SetUpBC_Advect;
    Physics_Advect_Bump *bump = &advect->sol.bump;
    bump->center[0] = 0.4;
    bump->center[1] = 0.2;
    bump->radius = 0.1;
    phys->maxspeed = 3.;       /* radius of mesh, kludge */
    /* Initial/transient solution with default boundary conditions */
    mod->solution = PhysicsSolution_Advect;
    mod->solutionctx = phys;CHKERRQ(ierr);
    mod->bcs[0] = mod->bcs[1] = mod->bcs[2] = DM_BOUNDARY_GHOSTED;
    PetscFunctionReturn(0);
}
/* End of model specific structure definition */

/* Defining the routine to set the initial value of the solution */
static PetscErrorCode
SolutionFunctional(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *modctx) {
    Model mod;
    PetscErrorCode ierr;
    PetscFunctionBegin;
    mod = (Model) modctx;
    ierr = (*mod->solution)(mod, time, x, u, mod->solutionctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode SetInitialCondition(DM dm, Vec X, User user) {
    PetscErrorCode(*func[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar * u, void *ctx);
    void *ctx[1];
    Model mod = user->model;
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    func[0] = SolutionFunctional;
    ctx[0] = (void *) mod;
    /* Passes the SolutionFunctional at time t = 0 (2nd arg) */
    ierr = DMProjectFunction(dm, 0.0, func, ctx, INSERT_ALL_VALUES, X);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* Initializing the TS object and defining the parameters */
static PetscErrorCode initializeTS(DM dm, User user, TS *ts) {
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = TSCreate(PetscObjectComm((PetscObject) dm), ts);CHKERRQ(ierr);
    ierr = TSSetType(*ts, TSSSP);CHKERRQ(ierr);
    ierr = TSSetDM(*ts, dm);CHKERRQ(ierr);
    ierr = TSMonitorSet(*ts, MyTSMonitor, PETSC_VIEWER_STDOUT_WORLD, NULL);CHKERRQ(ierr);
    ierr = DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, user);CHKERRQ(ierr);
    ierr = TSSetMaxTime(*ts, 2.0);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode MyTSMonitor(TS ts, PetscInt step, PetscReal ptime, Vec v, void *ctx) {
    PetscErrorCode ierr;
    PetscReal norm;
    MPI_Comm comm;
    PetscFunctionBeginUser;
    if (step < 0) PetscFunctionReturn(0); /* step of -1 indicates an interpolated solution */
    ierr = VecNorm(v, NORM_2, &norm);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "timestep %D time %g norm %g\n", step, (double) ptime, (double) norm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    MPI_Comm comm;
    PetscDS prob;
    PetscFV fvm;
    PetscLimiter limiter = NULL, noneLimiter = NULL;
    User user;
    Model mod;
    Physics phys;
    DM dm;
    PetscReal ftime, cfl, dt, minRadius;
    PetscInt dim, nsteps;
    TS ts;
    TSConvergedReason reason;
    Vec X;
    PetscBool simplex = PETSC_FALSE;
    PetscInt overlap;
    PetscErrorCode ierr;
    PetscMPIInt rank;
    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    if (ierr) return ierr;
    comm = PETSC_COMM_WORLD;
    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = PetscNew(&user);CHKERRQ(ierr);
    ierr = PetscNew(&user->model);CHKERRQ(ierr);
    ierr = PetscNew(&user->model->physics);CHKERRQ(ierr);
    mod = user->model;
    phys = mod->physics;
    mod->comm = comm;

    /* Setting a default cfl value */
    cfl = 0.9 * 4;

    /* Number of cells to overlap partitions */
    overlap = 1;

    /* Initializing the structures for the advection model */
    ierr = PhysicsCreate_Advect(mod, phys);CHKERRQ(ierr);

    /* Setting the number of fields and dof for the model */
    phys->dof = 1;

    /* Mesh creation routine */
    size_t i;
    for (i = 0; i < DIM; i++) {
        mod->bounds[2 * i] = 0.;
        mod->bounds[2 * i + 1] = 1.;
    }
    dim = 2;
    /* Defining the number of faces in each dimension */
    PetscInt cells[3] = {10, 10, 1};
    ierr = PetscOptionsGetInt(NULL, NULL, "-grid_x", &cells[0], NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-grid_y", &cells[1], NULL);CHKERRQ(ierr);
    ierr = DMPlexCreateBoxMesh(comm, dim, simplex, cells, NULL, NULL, mod->bcs, PETSC_TRUE, &dm);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-orig_dm_view");CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    /* set up BCs, functions, tags */
    ierr = DMCreateLabel(dm, "Face Sets");CHKERRQ(ierr);

    /* Configuring the DMPLEX object for FVM and distributing over all procs */
    DM dmDist;

    ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm = dmDist;
    }
    /* Constructing the ghost cells for DMPLEX object */
    DM gdm;
    ierr = DMPlexConstructGhostCells(dm, NULL, NULL, &gdm);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm = gdm;
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

    /* Creating and configuring the PetscFV object */
    ierr = PetscFVCreate(comm, &fvm);CHKERRQ(ierr);
    ierr = PetscFVSetFromOptions(fvm);CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvm, phys->dof);CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvm, dim);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fvm, "");CHKERRQ(ierr);
    /* Defining the component name for the PetscFV object */
    ierr = PetscFVSetComponentName(fvm, phys->dof, "q");CHKERRQ(ierr);
    /* Adding the field and specifying the dof (no. of components) */
    ierr = DMAddField(dm, NULL, (PetscObject) fvm);CHKERRQ(ierr);
    ierr = DMCreateDS(dm);CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, 0, user->model->physics->riemann);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 0, user->model->physics);CHKERRQ(ierr);
    ierr = (*mod->setupbc)(prob, phys);CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);

    ierr = initializeTS(dm, user, &ts);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(dm, &X);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) X, "solution");CHKERRQ(ierr);
    ierr = SetInitialCondition(dm, X, user);CHKERRQ(ierr);

    /* Setting the dt according to the speed and the smallest mesh width */
    ierr = DMPlexTSGetGeometryFVM(dm, NULL, NULL, &minRadius);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&phys->maxspeed, &mod->maxspeed, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) ts));CHKERRQ(ierr);
    dt = cfl * minRadius / mod->maxspeed;
    ierr = TSSetTimeStep(ts, dt);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    /* March the solution to the required final time */
    ierr = TSSolve(ts, X);CHKERRQ(ierr);
    ierr = TSGetSolveTime(ts, &ftime);CHKERRQ(ierr);
    ierr = TSGetStepNumber(ts, &nsteps);CHKERRQ(ierr);

    /* Get the reason for convergence (or divergence) */
    ierr = TSGetConvergedReason(ts, &reason);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %D steps\n", TSConvergedReasons[reason], (double) ftime, nsteps);CHKERRQ(ierr);

    /* Clean up routine */
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
    ierr = PetscFree(user->model->functionalMonitored);CHKERRQ(ierr);
    ierr = PetscFree(user->model->functionalCall);CHKERRQ(ierr);
    ierr = PetscFree(user->model->physics->data);CHKERRQ(ierr);
    ierr = PetscFree(user->model->physics);CHKERRQ(ierr);
    ierr = PetscFree(user->model);CHKERRQ(ierr);
    ierr = PetscFree(user);CHKERRQ(ierr);
    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = PetscLimiterDestroy(&limiter);CHKERRQ(ierr);
    ierr = PetscLimiterDestroy(&noneLimiter);CHKERRQ(ierr);
    ierr = PetscFVDestroy(&fvm);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
/* End of main() */

/*TEST

    test:
      args: -grid_x 10 -grid_y 10  -ts_max_steps 10

    test:
      suffix: 2
      args: -grid_x 10 -grid_y 10 -ts_type rk -ts_rk_type rk4

TEST*/
