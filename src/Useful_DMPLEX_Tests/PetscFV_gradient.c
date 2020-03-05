static char help[] = "Testing the gradient computations of a PetscFV object \n\n";

#include <petscdmplex.h>

#if defined(PETSC_HAVE_CGNS)
#undef I 
#include <cgnslib.h>
#endif
#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>
#endif

#include <petscksp.h>
#include <petscblaslapack.h>
#include <petscds.h>
#include <petscdmplex.h>
#include <petscdm.h>
#include <petscsnes.h>
#include <petscdmda.h>

/* Defining the application context */
typedef struct {
    PetscInt debug;             /* The debugging level */
    /* Domain and mesh definition */
    PetscInt dim;               /* The topological mesh dimension */
    PetscBool simplex;           /* Flag for simplex or tensor product mesh */
    PetscBool interpolate;       /* Generate intermediate mesh elements */
    PetscReal refinementLimit;   /* The largest allowable cell volume */
    /* Element definition */
    PetscInt qorder;            /* Order of the quadrature */
    PetscInt numComponents;     /* Number of field components */
    PetscFE fe;                /* The finite element */
    /* Testing space */
    PetscBool testFVgrad;        /* Test finite difference gradient routine */
    PetscReal constants[3];      /* Constant values for each dimension */
} AppCtx;

/* Defining/processing the options in the application context */
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
    PetscInt n = 3;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    options->debug = 0;
    options->dim = 2;
    options->simplex = PETSC_TRUE;
    options->interpolate = PETSC_TRUE;
    options->refinementLimit = 0.0;
    options->qorder = 0;
    options->numComponents = PETSC_DEFAULT;
    options->testFVgrad = PETSC_FALSE;
    options->constants[0] = 1.0;
    options->constants[1] = 2.0;
    options->constants[2] = 3.0;

    ierr = PetscOptionsBegin(comm, "", "Projection Test Options", "DMPlex");
            CHKERRQ(ierr);
            ierr = PetscOptionsBoundedInt("-debug", "The debugging level", "ex3.c", options->debug, &options->debug,
                                          NULL, 0);
            CHKERRQ(ierr);
            ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex3.c", options->dim, &options->dim,
                                        NULL, 1, 3);
            CHKERRQ(ierr);
            ierr = PetscOptionsBool("-simplex", "Flag for simplices or hexhedra", "ex3.c", options->simplex,
                                    &options->simplex, NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex3.c",
                                    options->interpolate, &options->interpolate, NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex3.c",
                                    options->refinementLimit, &options->refinementLimit, NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsBoundedInt("-qorder", "The quadrature order", "ex3.c", options->qorder, &options->qorder,
                                          NULL, 0);
            CHKERRQ(ierr);
            ierr = PetscOptionsBoundedInt("-num_comp", "The number of field components", "ex3.c",
                                          options->numComponents, &options->numComponents, NULL, PETSC_DEFAULT);
            CHKERRQ(ierr);
            ierr = PetscOptionsBool("-test_fv_grad", "Test finite volume gradient reconstruction", "ex3.c",
                                    options->testFVgrad, &options->testFVgrad, NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsRealArray("-constants", "Set the constant values", "ex3.c", options->constants, &n,
                                         NULL);
            CHKERRQ(ierr);
            ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    options->numComponents = options->numComponents < 0 ? options->dim : options->numComponents;

    PetscFunctionReturn(0);
}

/* Creating the mesh using DMPLEXCreateBoxMesh with the appropriate options */
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm) {
    PetscInt dim = user->dim;
    PetscBool isPlex;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;

    /* Defining the cell refinement in all of the dimensions */
    PetscInt cells[3] = {2, 2, 2};
    ierr = PetscOptionsGetInt(NULL, NULL, "-da_grid_x", &cells[0], NULL);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-da_grid_y", &cells[1], NULL);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-da_grid_z", &cells[2], NULL);
    CHKERRQ(ierr);
    ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, cells, NULL, NULL, NULL, PETSC_TRUE, dm);
    CHKERRQ(ierr);


    ierr = PetscObjectTypeCompare((PetscObject) *dm, DMPLEX, &isPlex);
    CHKERRQ(ierr);

    /* Partitioning the mesh in case of np > 1 */
    if (isPlex) {
        PetscPartitioner part;
        DM distributedMesh = NULL;

        /* Setting an uniform refinement across the box mesh */
        ierr = DMPlexSetRefinementUniform(*dm, PETSC_TRUE);
        CHKERRQ(ierr);

        /* Distribute mesh over processes */
        ierr = DMPlexGetPartitioner(*dm, &part);
        CHKERRQ(ierr);
        ierr = PetscPartitionerSetFromOptions(part);
        CHKERRQ(ierr);
        ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);
        CHKERRQ(ierr);
        if (distributedMesh) {
            ierr = DMDestroy(dm);
            CHKERRQ(ierr);
            *dm = distributedMesh;
        }
        ierr = PetscObjectSetName((PetscObject) *dm, "Hexahedral Mesh");
        CHKERRQ(ierr);
    }

    ierr = DMSetFromOptions(*dm);
    CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-dm_view");
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* Defining the PetscSection and also specifying parameters for the DA */
static PetscErrorCode SetupSection(DM dm, AppCtx *user) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    PetscSection sectionCell;
    PetscInt cStart, cEnd, c;

    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionCell);
    CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    CHKERRQ(ierr);
    ierr = PetscSectionSetChart(sectionCell, cStart, cEnd);
    CHKERRQ(ierr);

    /* Setting the number of fields in the DM
     * Typically set it to 1, and then set the components as the DOF. */
    ierr = DMSetNumFields(dm, 1);
    CHKERRQ(ierr);
    ierr = DMSetField(dm, 0, NULL, (PetscObject) user->fe);
    CHKERRQ(ierr);

    /* Setting the DOF for the Section being created */
    for (c = cStart; c < cEnd; ++c) {
        ierr = PetscSectionSetDof(sectionCell, c, user->numComponents);
        CHKERRQ(ierr);
    }

    ierr = PetscSectionSetUp(sectionCell);
    CHKERRQ(ierr);
    /* Setting it as the current section to be used. */
    ierr = DMSetLocalSection(dm, sectionCell);
    CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&sectionCell);
    CHKERRQ(ierr);
    /* Creating the Discretisation for the DM */
    ierr = DMCreateDS(dm);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode TestFVGrad(DM dm, AppCtx *user) {
    MPI_Comm comm;
    DM dmRedist, dmfv, dmgrad, dmCell;
    PetscFV fv;
    PetscInt cStart, cEnd, cEndInterior;
    PetscMPIInt size;
    Vec cellgeom, grad, locGrad;
    const PetscScalar *cgeom;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    comm = PetscObjectComm((PetscObject) dm);

    /* Configuring the adjacency functionality for the FVM problem */
    ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size);
    CHKERRQ(ierr);
    dmRedist = NULL;
    if (size > 1) {
        ierr = DMPlexDistributeOverlap(dm, 1, NULL, &dmRedist);
        CHKERRQ(ierr);
    }
    if (!dmRedist) {
        dmRedist = dm;
        ierr = PetscObjectReference((PetscObject) dmRedist);
        CHKERRQ(ierr);
    }

    /* Creating and configuring the PetscFV object */
    ierr = PetscFVCreate(comm, &fv);
    CHKERRQ(ierr);
    ierr = PetscFVSetType(fv, PETSCFVLEASTSQUARES);
    CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fv, user->numComponents);
    CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fv, user->dim);
    CHKERRQ(ierr);

    /* Setting the limiters in PetscFV */
    PetscLimiter limiter = NULL, limiterType = NULL;
    ierr = PetscFVGetLimiter(fv, &limiter);
    CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) limiter);
    CHKERRQ(ierr);
    ierr = PetscLimiterCreate(PetscObjectComm((PetscObject) fv), &limiterType);
    CHKERRQ(ierr);
    ierr = PetscLimiterSetType(limiterType, PETSCLIMITERMINMOD);
    CHKERRQ(ierr);
    ierr = PetscFVSetLimiter(fv, limiterType);
    CHKERRQ(ierr);

    ierr = PetscFVSetFromOptions(fv);
    CHKERRQ(ierr);
    ierr = PetscFVSetUp(fv);
    CHKERRQ(ierr);

    /* Constructing the Ghost cells in the PLEX mesh */
    ierr = DMPlexConstructGhostCells(dmRedist, NULL, NULL, &dmfv);
    CHKERRQ(ierr);
    ierr = DMDestroy(&dmRedist);
    CHKERRQ(ierr);
    ierr = DMSetNumFields(dmfv, 1);
    CHKERRQ(ierr);
    ierr = DMSetField(dmfv, 0, NULL, (PetscObject) fv);
    CHKERRQ(ierr);
    ierr = DMCreateDS(dmfv);
    CHKERRQ(ierr);

    /* Obtaining the Gradient DM object from SNES */
    ierr = DMPlexSNESGetGradientDM(dmfv, fv, &dmgrad);
    CHKERRQ(ierr);

    ierr = DMPlexGetHeightStratum(dmfv, 0, &cStart, &cEnd);
    CHKERRQ(ierr);
    ierr = DMPlexSNESGetGeometryFVM(dmfv, NULL, &cellgeom, NULL);
    CHKERRQ(ierr);

    ierr = VecGetDM(cellgeom, &dmCell);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellgeom, &cgeom);
    CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dmgrad, &grad);
    CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmgrad, &locGrad);
    CHKERRQ(ierr);

    ierr = DMPlexGetInteriorCellStratum(dmgrad, NULL, &cEndInterior);
    CHKERRQ(ierr);
    cEndInterior = (cEndInterior < 0) ? cEnd : cEndInterior;

    /* Setting the initial condition */
    Vec locX;
    PetscInt c;
    PetscScalar *x;
    const PetscScalar *gradArray;

    ierr = DMGetLocalVector(dmfv, &locX);
    CHKERRQ(ierr);
    ierr = VecGetArray(locX, &x);
    CHKERRQ(ierr);

    /* Here, we initialize the vector X with a given function. */
    for (c = cStart; c < cEnd; c++) {
        PetscFVCellGeom *cg;
        PetscScalar *xc;
        PetscReal r;

        /* This function returns an array "xc" when given the array "x" and the current position
         * x - the array returned by VecGetArray().
         * xc - array of size = dof (Here number of components = dof = 2)*/
        ierr = DMPlexPointLocalRef(dmfv, c, x, &xc);
        CHKERRQ(ierr);

        ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);
        CHKERRQ(ierr);

        r = PetscSqrtReal(
                (cg->centroid[0] - .5) * (cg->centroid[0] - .5) + (cg->centroid[1] - .5) * (cg->centroid[1] - .5));

        /* The array xc is defined as per its degree of freedom. */
        if (r <= 0.350) {
            xc[0] = PetscExpReal(-30.0 * r * r * r);
            xc[1] = PetscExpReal(-10.0 * r * r * r);
        } else {
            xc[0] = 0.0;
            xc[1] = 0.0;
        }
        printf("cell %d | 1 : %f | 2 : %f \n", c, xc[0], xc[1]);

    }

    /* Asking DMPlex to compute the gradients given the PetscFV object and local vector */
    ierr = DMPlexReconstructGradientsFVM(dmfv, locX, grad);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmgrad, grad, INSERT_VALUES, locGrad);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmgrad, grad, INSERT_VALUES, locGrad);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGrad, &gradArray);
    CHKERRQ(ierr);

    /* Here the computed gradients are printed according the cell
     *
     * Note that the loop spans only across the interior cells.
     * This is as a result of the gradients being relevant in the interior. */
    for (c = cStart; c < cEndInterior; c++) {
        PetscScalar *compGrad;

        ierr = DMPlexPointLocalRead(dmgrad, c, gradArray, &compGrad);
        CHKERRQ(ierr);

        printf("Component 1: Cell %d : Gradient x = %f, Gradient y = %f \n", c, compGrad[0], compGrad[1]);
        printf("Component 2: Cell %d : Gradient x = %f, Gradient y = %f \n", c, compGrad[2], compGrad[3]);

    }

    /* Clean up routine */
    ierr = VecRestoreArrayRead(locGrad, &gradArray);
    CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmfv, &locX);
    CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmgrad, &locGrad);
    CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmgrad, &grad);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellgeom, &cgeom);
    CHKERRQ(ierr);
    ierr = DMDestroy(&dmfv);
    CHKERRQ(ierr);
    ierr = PetscFVDestroy(&fv);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    DM dm;
    AppCtx user;
    PetscErrorCode ierr;

    /* Intitialization routine */
    ierr = PetscInitialize(&argc, &argv, NULL, help);
    if (ierr) return ierr;
    ierr = ProcessOptions(PETSC_COMM_WORLD, &user);
    CHKERRQ(ierr);
    ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);
    CHKERRQ(ierr);
    ierr = PetscFECreateDefault(PETSC_COMM_WORLD, user.dim, user.numComponents, user.simplex, NULL, user.qorder,
                                &user.fe);
    CHKERRQ(ierr);
    ierr = SetupSection(dm, &user);
    CHKERRQ(ierr);

    /* Testing the Gradient computations */
    if (user.testFVgrad) {
        ierr = TestFVGrad(dm, &user);
        CHKERRQ(ierr);
    }

    /* Clean up routine */
    ierr = PetscFEDestroy(&user.fe);
    CHKERRQ(ierr);
    ierr = DMDestroy(&dm);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}

/*
test:
suffix: Conforming tensor with no limitors
args: -test_fv_grad -petsclimiter_type none -petscpartitioner_type simple -tree -simplex 0 -dim 2 -num_comp 2
test:
suffix: Conforming tensor with limitors used
args: -test_fv_grad -petsclimiter_type none -petscpartitioner_type simple -tree -simplex 0 -dim 3 -num_comp 3
 */
