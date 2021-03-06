static char help[] = "Simple Advection-diffusion equation solved using FVM in DMPLEX\n";

/*
   Solves the simple advection equation given by

   q_t + u (q_x) + v (q_y) - D (q_xx + q_yy) = 0 using FVM and First Order Upwind discretization. 

   with a user defined initial condition.

   with dirichlet/neumann conditions on the four boundaries of the domain.

   User can define the mesh parameters either in the command line or inside
   the ProcessOptions() routine.

*/

#include <petscdmplex.h>
#include <petscts.h>
#include <petscblaslapack.h>

#if defined(PETSC_HAVE_CGNS)
#undef I
#include <cgnslib.h>
#endif
/*
   User-defined routines
*/
extern PetscErrorCode FormFunction(TS, PetscReal, Vec, Vec, void *), FormInitialSolution(DM, Vec);
extern PetscErrorCode MyTSMonitor(TS, PetscInt, PetscReal, Vec, void *);
extern PetscErrorCode MySNESMonitor(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);

/* Defining the usr defined context */
typedef struct {
    DM da;
    PetscBool interpolate;                  /* Generate intermediate mesh elements */
    char filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
    PetscInt dim;
    PetscErrorCode (**bcFuncs)(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    PetscScalar u, v, diffusion;
    PetscScalar delta_x, delta_y;
    PetscInt cells[2];
} AppCtx;

/* Options for the scenario */
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    options->interpolate = PETSC_TRUE;
    options->filename[0] = '\0';
    options->dim = 2;
    options->bcFuncs = NULL;
    options->u = 2.5;
    options->v = 0.0;
    options->cells[0] = 20;
    options->cells[1] = 20;
    options->diffusion = 0.0;

    ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "advection_DMPLEX.c",options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-filename", "The mesh file", "advection_DMPLEX.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The dimension of problem used for non-file mesh", "advection_DMPLEX.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-u", "The x component of the convective coefficient", "advection_DMPLEX.c", options->u, &options->u, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-v", "The y component of the convective coefficient", "advection_DMPLEX.c", options->v, &options->v, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-diffus", "The diffusive coefficient", "advection_DMPLEX.c", options->diffusion, &options->diffusion, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    PetscFunctionReturn(0);
}

/*
  User can provide the file containing the mesh.
  Or can generate the mesh using DMPlexCreateBoxMesh with the specified options.
*/
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm) {
    size_t len;
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = PetscStrlen(user->filename, &len);CHKERRQ(ierr);
    if (!len) {
        DMLabel label;
        ierr = DMPlexCreateBoxMesh(comm, user->dim, PETSC_FALSE, user->cells, NULL, NULL, NULL, user->interpolate, dm);CHKERRQ(ierr);
        /* Mark boundary and set BC */
        ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
        ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
        ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
        ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
        ierr = PetscMalloc1(1, &user->bcFuncs);CHKERRQ(ierr);
    } else {
        ierr = DMPlexCreateFromFile(comm, user->filename, user->interpolate, dm);CHKERRQ(ierr);
    }
    ierr = PetscObjectSetName((PetscObject) * dm, "Mesh");CHKERRQ(ierr);
    ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* Checking the Mesh Structure (Simplex or Tensor) */
static PetscErrorCode CheckMeshTopology(DM dm) {
    PetscInt coneSize, cStart, dim;
    PetscBool isSimplex;
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    /* Get starting and ending point (here NULL) for needed height */
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
    /* Possibly for a triangle for a 2D mesh? */
    isSimplex = coneSize == dim + 1 ? PETSC_TRUE : PETSC_FALSE;
    ierr = DMPlexCheckSymmetry(dm);CHKERRQ(ierr);
    ierr = DMPlexCheckSkeleton(dm, 0);CHKERRQ(ierr);
    ierr = DMPlexCheckFaces(dm, 0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


  /* Subroutine to define faces information and corresponding neighbors */
static PetscErrorCode CheckMeshGeometry(DM dm) {
    PetscInt dim, coneSize, cStart, cEnd, c; //cStart, cEnd - cells
    PetscReal *v0, *J, *invJ, detJ;
    PetscInt conesize;
    const PetscInt *cone;
    PetscInt nC;
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
    ierr = PetscMalloc3(dim, &v0, dim * dim, &J, dim * dim, &invJ);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
        ierr = DMPlexGetConeSize(dm, c, &conesize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
        if (detJ <= 0.0)
            SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for cell %D", (double) detJ, c);
    }
    ierr = PetscFree3(v0, J, invJ);CHKERRQ(ierr);

    for (c = cStart; c < cEnd; ++c) {
        const PetscInt *faces;
        PetscInt numFaces, f;
        if ((c < cStart) || (c >= cEnd))
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a cell", c);
        DMPlexGetConeSize(dm, c, &numFaces);
        DMPlexGetCone(dm, c, &faces);
        for (f = 0; f < numFaces; ++f) {
            const PetscInt face = faces[f];
            const PetscInt *neighbors;
            DMPlexGetSupportSize(dm, face, &nC);
            /* Check for the boundary faces possibly */
            if (nC != 2) continue;
            DMPlexGetSupport(dm, face, &neighbors);
        }
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode PostStep(TS ts)
{
    PetscErrorCode ierr;
    PetscInt stepi;
    Vec X;
    PetscReal time;
    DM dm;
    PetscFunctionBegin;
    ierr = TSGetSolution(ts, &X);CHKERRQ(ierr);
    ierr = VecGetDM(X, &dm);CHKERRQ(ierr);
    ierr = DMGetOutputSequenceNumber(dm, &stepi, NULL);CHKERRQ(ierr);
    ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
    ierr = DMSetOutputSequenceNumber(dm, stepi + 1, time);CHKERRQ(ierr); // stay ahead of initial solution
    ierr = VecViewFromOptions(X, NULL, "-vec_view");CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

    /* This routine is responsible for defining the local solution vector x
    with a given initial solution.
    The initial solution can be modified accordingly inside the loops.
    No need for a local vector because there is exchange of information
    across the processors. Unlike for FormFunction which depends on the neighbours */
PetscErrorCode FormInitialSolution(DM da, Vec U) {
    PetscErrorCode ierr;
    PetscScalar *u;
    PetscFunctionBeginUser;
    /* Get pointers to vector data */
    ierr = VecGetArray(U, &u);CHKERRQ(ierr);
    /* Get local grid boundaries */
    PetscInt cell, cStart, cEnd;
    PetscReal cellvol, centroid[3], normal[3];
    ierr = DMPlexGetHeightStratum(da, 0, &cStart, &cEnd);CHKERRQ(ierr);
    /* Assigning the values at the cell centers based on x and y directions */
    for (cell = cStart; cell < cEnd; cell++) {
        DMPlexComputeCellGeometryFVM(da, cell, &cellvol, centroid, normal);
        if (centroid[0] > 0.9 && centroid[0] < 0.95) {
            if (centroid[1] > 0.9 && centroid[1] < 0.95) u[cell] = 2.0;
        }
        else u[cell] = 0;
    }
    ierr = VecRestoreArray(U, &u);CHKERRQ(ierr);
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

/*
   MySNESMonitor - illustrate how to set user-defined monitoring routine for SNES.
   Input Parameters:
     snes - the SNES context
     its - iteration number
     fnorm - 2-norm function value (may be estimated)
     ctx - optional user-defined context for private data for the
         monitor routine, as set by SNESMonitorSet()
*/
PetscErrorCode MySNESMonitor(SNES snes, PetscInt its, PetscReal fnorm, PetscViewerAndFormat *vf) {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = SNESMonitorDefaultShort(snes, its, fnorm, vf);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  ts - the TS context
.  X - input vector
.  ctx - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode FormFunction(TS ts, PetscReal ftime, Vec X, Vec F, void *ctx) {
    AppCtx *user = (AppCtx *) ctx;
    DM da = (DM) user->da;
    PetscErrorCode ierr;
    PetscScalar *x, *f;
    Vec localX;
    PetscInt fStart, fEnd, nF;
    PetscInt cell, cStart, cEnd, nC;
    /* Get the local vector from the DM object. */
    PetscFunctionBeginUser;
    ierr = DMGetLocalVector(da, &localX);CHKERRQ(ierr);

    /* Scatter ghost points to local vector,using the 2-step process
       DMGlobalToLocalBegin(),DMGlobalToLocalEnd(). */
    ierr = DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX);CHKERRQ(ierr);
    /* Get pointers to vector data. */
    ierr = VecGetArray(localX, &x);CHKERRQ(ierr);
    ierr = VecGetArray(F, &f);CHKERRQ(ierr);

    /* Obtaining local cell and face ownership */
    ierr = DMPlexGetHeightStratum(da, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(da, 1, &fStart, &fEnd);CHKERRQ(ierr);

    DM dmFace, gradDM;      /* DMPLEX for face geometry */
    PetscFV fvm;                /* specify type of FVM discretization */
    Vec cellGeom, faceGeom; /* vector of structs related to cell/face geometry*/
    const PetscScalar *fgeom;             /* values stored in the vector facegeom */
    PetscFVFaceGeom *fgA;               /* struct with face geometry information */

    /* Creating the PetscFV object to obtain face and cell geometry.
    Later to be used to compute face centroid to find cell widths. */

    ierr = PetscFVCreate(PETSC_COMM_WORLD, &fvm);CHKERRQ(ierr);
    ierr = PetscFVSetType(fvm, PETSCFVUPWIND);CHKERRQ(ierr);
    /*....Retrieve precomputed cell geometry....*/
    ierr = DMPlexGetDataFVM(da, fvm, &cellGeom, &faceGeom, &gradDM);CHKERRQ(ierr);
    ierr = VecGetDM(faceGeom, &dmFace);CHKERRQ(ierr);
    ierr = VecGetArrayRead(faceGeom, &fgeom);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Spanning through all the cells and an inner loop through the faces. Find the
    face neighbors and pick the upwinded cell value for flux.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    const PetscInt *cellcone, *cellsupport;
    PetscScalar flux_east, flux_west, flux_north, flux_south, flux_centre;
    PetscScalar centroid_x[2], centroid_y[2], boundary = 0.0;
    PetscScalar boundary_left = 0.0;
    PetscScalar u_plus, u_minus, v_plus, v_minus;
    PetscScalar delta_x, delta_y;

    u_plus = PetscMax(user->u, (PetscScalar) 0);
    u_minus = PetscMin(user->u, (PetscScalar) 0);
    v_plus = PetscMax(user->v, (PetscScalar) 0);
    v_minus = PetscMin(user->v, (PetscScalar) 0);

    for (cell = cStart; cell < cEnd; cell++) {
        /* Obtaining the faces of the cell */
        DMPlexGetConeSize(da, cell, &nF);
        DMPlexGetCone(da, cell, &cellcone);

        /* south */
        DMPlexPointLocalRead(dmFace, cellcone[0], fgeom, &fgA);
        centroid_y[0] = fgA->centroid[1];
        /* North */
        DMPlexPointLocalRead(dmFace, cellcone[2], fgeom, &fgA);
        centroid_y[1] = fgA->centroid[1];
        /* West */
        DMPlexPointLocalRead(dmFace, cellcone[3], fgeom, &fgA);
        centroid_x[0] = fgA->centroid[0];
        /* East */
        DMPlexPointLocalRead(dmFace, cellcone[1], fgeom, &fgA);
        centroid_x[1] = fgA->centroid[0];

        /* Computing the cell widths in the x and y direction */
        delta_x = centroid_x[1] - centroid_x[0];
        delta_y = centroid_y[1] - centroid_y[0];

        /* Getting the neighbors of each face
           Going through the faces by the order (cellcone) */

        /* cellcone[0] - south */
        DMPlexGetSupportSize(da, cellcone[0], &nC);
        DMPlexGetSupport(da, cellcone[0], &cellsupport);
        if (nC == 2) flux_south = (x[cellsupport[0]] * (-v_plus - user->diffusion * delta_x)) / delta_y;
        else flux_south = (boundary * (-v_plus - user->diffusion * delta_x)) / delta_y;

        /* cellcone[1] - east */
        DMPlexGetSupportSize(da, cellcone[1], &nC);
        DMPlexGetSupport(da, cellcone[1], &cellsupport);
        if (nC == 2) flux_east = (x[cellsupport[1]] * (u_minus - user->diffusion * delta_y)) / delta_x;
        else flux_east = (boundary * (u_minus - user->diffusion * delta_y)) / delta_x;

        /* cellcone[2] - north */
        DMPlexGetSupportSize(da, cellcone[2], &nC);
        DMPlexGetSupport(da, cellcone[2], &cellsupport);
        if (nC == 2) flux_north = (x[cellsupport[1]] * (v_minus - user->diffusion * delta_x)) / delta_y;
        else flux_north = (boundary * (v_minus - user->diffusion * delta_x)) / delta_y;

        /* cellcone[3] - west */
        DMPlexGetSupportSize(da, cellcone[3], &nC);
        DMPlexGetSupport(da, cellcone[3], &cellsupport);
        if (nC == 2) flux_west = (x[cellsupport[0]] * (-u_plus - user->diffusion * delta_y)) / delta_x;
        else flux_west = (boundary_left * (-u_plus - user->diffusion * delta_y)) / delta_x;

        /* Contribution by the cell to the fluxes */
        flux_centre = x[cell] * ((u_plus - u_minus + 2 * user->diffusion * delta_y) / delta_x +
                                 (v_plus - v_minus + 2 * user->diffusion * delta_x) / delta_y);

        /* Calculating the net flux for each cell
           and computing the RHS time derivative f[.] */
        f[cell] = -(flux_centre + flux_east + flux_west + flux_north + flux_south);

    }

    ierr = VecRestoreArray(localX, &x);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(F, &f);
    CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &localX);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    TS ts;                         /* time integrator */
    SNES snes;
    Vec x, r;                        /* solution, residual vectors */
    PetscErrorCode ierr;
    DM da;
    PetscMPIInt rank;
    PetscViewerAndFormat *vf;
    AppCtx user;                             /* mesh context */

    /* Initialize program */
    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    if (ierr) return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
    /* Create distributed array (DMPLEX) to manage parallel grid and vectors */
    ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
    ierr = CreateMesh(PETSC_COMM_WORLD, &user, &da);CHKERRQ(ierr);
    ierr = CheckMeshTopology(da);CHKERRQ(ierr);
    ierr = CheckMeshGeometry(da);CHKERRQ(ierr);

    /* Specifying the fields and dof for the formula through PETSc Section
    Create a scalar field u with 1 component on cells, faces and edges. */
    PetscInt numFields = 1, numBC, i;
    PetscInt numComp[1];
    PetscInt numDof[numFields * (user.dim + 1)];
    PetscInt bcField[1];
    PetscSection section;
    IS bcPointIS[1];

    numComp[0] = 1;

    for (i = 0; i < numFields * (user.dim + 1); ++i) numDof[i] = 0;

    numDof[0 * (user.dim + 1)] = 1;
    numDof[0 * (user.dim + 1) + user.dim - 1] = 1;
    numDof[0 * (user.dim + 1) + user.dim] = 1;

    /* Setup boundary conditions */
    numBC = 1;
    /* Prescribe a Dirichlet condition on u on the boundary
       Label "marker" is made by the mesh creation routine  */
    bcField[0] = 0;
    ierr = DMGetStratumIS(da, "marker", 1, &bcPointIS[0]);CHKERRQ(ierr);

    /* Create a PetscSection with this data layout */
    ierr = DMSetNumFields(da, numFields);CHKERRQ(ierr);
    ierr = DMPlexCreateSection(da, NULL, numComp, numDof, numBC, bcField, NULL, bcPointIS, NULL, &section);CHKERRQ(ierr);
    ierr = ISDestroy(&bcPointIS[0]);CHKERRQ(ierr);

    /* Name the Field variables */
    ierr = PetscSectionSetFieldName(section, 0, "u");CHKERRQ(ierr);

    /* Tell the DM to use this section (with the specified fields and dof) */
    ierr = DMSetLocalSection(da, section);CHKERRQ(ierr);
    user.da = da;

    /* Extract global vectors from DMDA; then duplicate for remaining
       vectors that are the same types */

    /* Create a Vec with this layout and view it */
    ierr = DMGetGlobalVector(da, &x);CHKERRQ(ierr);
    ierr = VecDuplicate(x, &r);CHKERRQ(ierr);

    /* Create timestepping solver context */
    ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
    ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts, NULL, FormFunction, &user);CHKERRQ(ierr);

    ierr = TSSetMaxTime(ts, 1.0);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts, MyTSMonitor, PETSC_VIEWER_STDOUT_WORLD, NULL);CHKERRQ(ierr);
    ierr = TSSetDM(ts, da);CHKERRQ(ierr);

    /* Customize nonlinear solver */
    ierr = TSSetType(ts, TSEULER);CHKERRQ(ierr);
    ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
    ierr = PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, &vf);CHKERRQ(ierr);
    ierr = SNESMonitorSet(snes, (PetscErrorCode (*)(SNES, PetscInt, PetscReal, void *)) MySNESMonitor, vf,(PetscErrorCode (*)(void **)) PetscViewerAndFormatDestroy);CHKERRQ(ierr);

     /* Set initial conditions */
    ierr = FormInitialSolution(da, x);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, .0001);CHKERRQ(ierr);
    ierr = TSSetSolution(ts, x);CHKERRQ(ierr);
    /* Set runtime options */
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    /* Solve nonlinear system */
    ierr = TSSetPostStep(ts, PostStep);CHKERRQ(ierr);
    ierr = PostStep(ts);CHKERRQ(ierr);
    ierr = TSSolve(ts, x);CHKERRQ(ierr);

    /* Clean up routine */
    ierr = DMRestoreGlobalVector(da, &x);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}

/*TEST

    test:
      args: -ts_max_steps 5 -draw_pause 10

    test:
      suffix: 2
      args: -ts_max_steps 5  -u 1.0 -v 0.0 -diffus 1.0 -draw_pause 10

    test:
      suffix: 3
      args: -ts_max_steps 5 -ts_type glle -u 1.0 -v 0.0 -diffus 1.0 -draw_pause 10

TEST*/
