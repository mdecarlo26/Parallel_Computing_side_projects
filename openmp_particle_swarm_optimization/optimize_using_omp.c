/* Implementation of PSO using OpenMP.
 *
 * Author: Naga Kandasamy
 * Date: February 23, 2024
 *
 */
#define _POSIX_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "pso.h"
#include "omp.h"
#include <sys/time.h>

swarm_t *pso_init_omp(char *function, int dim, int swarm_size, 
                  float xmin, float xmax,int threads_cnt)
{
    int i, j, g;
    int status;
    float fitness;
    swarm_t *swarm;
    particle_t *particle;
    float best_fitness = INFINITY;
    float local_fitness= INFINITY;
    int local_g;
    int l;

    swarm = (swarm_t *)malloc(sizeof(swarm_t));
    swarm->num_particles = swarm_size;
    swarm->particle = (particle_t *)malloc(swarm_size * sizeof(particle_t));
    if (swarm->particle == NULL)
        exit(EXIT_FAILURE);
#pragma omp parallel private(i,j,fitness,status,particle,local_g,l) shared(swarm,g,best_fitness) num_threads(threads_cnt) firstprivate(local_fitness)
{
#pragma omp for
    for (i = 0; i < swarm->num_particles; i++) {
        particle = &swarm->particle[i];
        particle->dim = dim; 
        /* Generate random particle position */
        particle->x = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
           particle->x[j] = uniform(xmin, xmax);

       /* Generate random particle velocity */ 
        particle->v = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
            particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));

        /* Initialize best position for particle */
        particle->pbest = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
            particle->pbest[j] = particle->x[j];

        /* Initialize particle fitness */
        
        status = pso_eval_fitness(function, particle, &fitness);
        if (status < 0) {
            fprintf(stderr, "Could not evaluate fitness. Unknown function provided.\n");
            exit(EXIT_FAILURE);
        }
        particle->fitness = fitness;

        /* Initialize index of best performing particle */
        particle->g = -1;
        
    }
    /* Get index of particle with best fitness */
    //inline get best fitness function
#pragma omp for private(l)
        for (l = 0; l < swarm->num_particles; l++) {
            particle = &swarm->particle[l];
            if (particle->fitness < local_fitness) {   
                local_fitness = particle->fitness;
                local_g = l;
#pragma omp critical
{
                if(local_fitness< best_fitness){
                // printf("P num: %d, local %f, best: %f\n",i,local_fitness,best_fitness);
                    best_fitness = local_fitness;
                    g = local_g;
                }
}
        }
    }//implicit barrier

#pragma omp for private(i)
    for (i = 0; i < swarm->num_particles; i++) {
        particle = &swarm->particle[i];
        particle->g = g;
    }
}//implicit barrier
    return swarm;
}


int pso_solve_omp(char *function, swarm_t *swarm, float xmax, float xmin, int max_iter,int thread_cnt)
{
    int i, j, iter, g,l;
    float w, c1, c2;
    float r1, r2;
    float curr_fitness;
    particle_t *particle, *gbest;

    float best_fitness = INFINITY;
    float local_fitness= INFINITY;
    int local_g;

    w = 0.79;
    c1 = 1.49;
    c2 = 1.49;
    iter = 0;
    g = -1;
    unsigned int seed = time(NULL); // Seed the random number generator 
#pragma omp parallel num_threads(thread_cnt) private(i, j, r1, r2, particle, curr_fitness, gbest,local_g) shared(g,best_fitness) firstprivate(local_fitness)
{
    while (iter < max_iter) {
#pragma omp for private(gbest,particle,i)
        for (i = 0; i < swarm->num_particles; i++) {
            particle = &swarm->particle[i];
            gbest = &swarm->particle[particle->g];  /* Best performing particle from last iteration */ 
            for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
                r1 = (float)rand_r(&seed)/(float)RAND_MAX;
                r2 = (float)rand_r(&seed)/(float)RAND_MAX;
                /* Update particle velocity */
                particle->v[j] = w * particle->v[j]\
                                 + c1 * r1 * (particle->pbest[j] - particle->x[j])\
                                 + c2 * r2 * (gbest->x[j] - particle->x[j]);
                /* Clamp velocity */
                if ((particle->v[j] < -fabsf(xmax - xmin)) || (particle->v[j] > fabsf(xmax - xmin))) 
                    particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));

                /* Update particle position */
                particle->x[j] = particle->x[j] + particle->v[j];
                if (particle->x[j] > xmax)
                    particle->x[j] = xmax;
                if (particle->x[j] < xmin)
                    particle->x[j] = xmin;
            } /* State update */
            
            /* Evaluate current fitness */
            pso_eval_fitness(function, particle, &curr_fitness);

            /* Update pbest */
            if (curr_fitness < particle->fitness) {
                particle->fitness = curr_fitness;
                for (j = 0; j < particle->dim; j++)
                    particle->pbest[j] = particle->x[j];
            }
        } /* Particle loop */ //impolicit barrier

        /* Identify best performing particle */
        //inline get best fitness function
#pragma omp for private(l)
        for (l = 0; l < swarm->num_particles; l++) {
            particle = &swarm->particle[l];
            if (particle->fitness < local_fitness) {   
                local_fitness = particle->fitness;
                local_g = l;
#pragma omp critical
{
                if(local_fitness< best_fitness){
                // printf("P num: %d, local %f, best: %f\n",i,local_fitness,best_fitness);
                    best_fitness = local_fitness;
                    g = local_g;
                }
}
        }
    }//implicit barrier

        
#pragma omp for private(i)
        for (i = 0; i < swarm->num_particles; i++) {
            particle = &swarm->particle[i];
            particle->g = g;
        }//implicit barrier

#ifdef SIMPLE_DEBUG
        /* Print best performing particle */
        fprintf(stderr, "\nIteration %d:\n", iter);
        pso_print_particle(&swarm->particle[g]);
#endif
#pragma omp single
{
        iter++;
}//implicit barrier
    } /* End of iteration */
}//implicit barrier
    return g;
}

int optimize_using_omp(char *function, int dim, int swarm_size, 
                       float xmin, float xmax, int num_iter, int num_threads)
{
    {
     /* Initialize PSO */
    swarm_t *swarm;
    srand(time(NULL));
    swarm = pso_init_omp(function, dim, swarm_size, xmin, xmax,num_threads);
    if (swarm == NULL) {
        fprintf(stderr, "Unable to initialize PSO\n");
        exit(EXIT_FAILURE);
    }

#ifdef VERBOSE_DEBUG
    pso_print_swarm(swarm);
#endif

    /* Solve PSO */
    int g; 
    g = pso_solve_omp(function, swarm, xmax, xmin, num_iter,num_threads);
    if (g >= 0) {
        fprintf(stderr, "Solution:\n");
        pso_print_particle(&swarm->particle[g]);
    }

    pso_free(swarm);
    return g;
}
}
