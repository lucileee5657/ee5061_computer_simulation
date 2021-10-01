/* External definitions for job-shop model. */

#include "simlib.h"              /* Required for use of simlib.c. */

#define EVENT_ARRIVAL         1  /* Event type for arrival of a job to the
                                    system. */
#define EVENT_DEPARTURE       2  /* Event type for departure of a job from a
                                    particular station. */
#define EVENT_END_SIMULATION  3  /* Event type for end of the simulation. */
#define STREAM_INTERARRIVAL   1  /* Random-number stream for interarrivals. */
#define STREAM_JOB_TYPE       2  /* Random-number stream for job types. */
#define STREAM_SERVICE        3  /* Random-number stream for service times. */
#define MAX_NUM_STATIONS      10  /* Maximum number of stations. */
#define MAX_NUM_JOB_TYPES     10  /* Maximum number of job types. */

/* Declare non-simlib global variables. */

int   num_servers,                      //For N: number of edge servers
      num_job_types,                    //each type for each server
      num_vms[MAX_NUM_STATIONS + 1],    //number of vms at each edge server
      //num_tasks[MAX_NUM_JOB_TYPES +1],
      //route[MAX_NUM_JOB_TYPES +1][MAX_NUM_STATIONS + 1],
      i, j, job_type, task, num_machines_busy[MAX_NUM_STATIONS + 1];

float mean_interarrival,                //different arrival rate at each edge server
      length_simulation,
      prob_distrib_job_type[26],                       
      mean_service;                     //departure rate at edge device
      //mean_service[MAX_NUM_JOB_TYPES +1][ MAX_NUM_STATIONS + 1];
FILE  *infile, *outfile, *outfiles;

/* Declare non-simlib functions. */

int   one_setting();
void  arrive(int new_job);
void  depart(void);
void  report(void);




int main()
{
    
    infile  = fopen("terms.in",  "r");
    outfile = fopen("term.out", "w");
    outfiles = fopen("terms.out", "w");
    
    fscanf(infile, "%d %d %f %f", &num_servers, &num_job_types,
           &mean_service, &length_simulation);
    for (j = 1; j <= num_servers; ++j)
        fscanf(infile, "%d", &num_vms[j]);
    
    for (i = 1; i <= num_job_types; ++i)
        fscanf(infile, "%f", &prob_distrib_job_type[i]);
    
    
    for (int lambda = 1; lambda <= 10; ++lambda){
        mean_interarrival = 1.0/10/lambda;
        fprintf(outfile, "\n\n\nEdge Server Model (lambda = %d, original)\n\n", lambda);
        fprintf(outfiles, "%d, 0", lambda);
        init_simlib();
        one_setting();
    }
    
    
    float alpha[5];
    alpha[0] = 1.0;
    alpha[1] = 0.7;
    alpha[2] = 0.5;
    alpha[3] = 0.3;
    alpha[4] = 0.1;

    for (int lambda = 1; lambda <= 10; ++lambda){
        mean_interarrival = 1.0/10/lambda;
        
        for (int a = 1; a <= 5; ++a){
            fscanf(infile, "%d", &num_servers);
            num_job_types = num_servers;
            for (int x = 1; x <= 26; ++x)
                prob_distrib_job_type[x] = 0.0;
            for (int x = 1; x <= num_job_types; ++x)
                fscanf(infile, "%f", &prob_distrib_job_type[x]);
            fprintf(outfile, "\n\n\n*** Edge Server Model (lambda = %d, alpha = %.2f)\n\n", lambda, alpha[a-1]);
            fprintf(outfiles, "%d, %.2f", lambda, alpha[a-1]);
            init_simlib();
            one_setting();
        }
        
    }
    fclose(infile);
    fclose(outfile);

    return 0;

}

int one_setting()  /* Main function. */
{
    /* Open input and output files. */

    //infile  = fopen("term.in",  "r");
    //outfile = fopen("term.out", "w");

    /* Read input parameters. */
    /*
    fscanf(infile, "%d %d %f %f", &num_servers, &num_job_types,
           &mean_service, &length_simulation);
    for (j = 1; j <= num_servers; ++j)
        fscanf(infile, "%d", &num_vms[j]);
    fscanf(infile, "%f", &mean_interarrival);
    for (i = 1; i <= num_job_types; ++i)
        fscanf(infile, "%f", &prob_distrib_job_type[i]);
    */

    /* Write report heading and input parameters. */

    //fprintf(outfile, "Edge Server Model\n\n");
    fprintf(outfile, "Number of edge servers%21d\n\n", num_servers);
    fprintf(outfile, "Number of virtual machines in each server     ");
    for (j = 1; j <= num_servers; ++j)
        fprintf(outfile, "%5d", num_vms[j]);
    fprintf(outfile, "\n\nNumber of job types%25d\n\n", num_job_types);
    
    fprintf(outfile, "\n\nMean interarrival time of jobs%14.4f sec in total\n\n",
            mean_interarrival);
    fprintf(outfile, "Length of the simulation%20.1f minutes\n\n\n",
            length_simulation);
    fprintf(outfile, "\n\nMean departure time of jobs%16.2f sec in total\n\n",
            mean_service);
    
    for (i = 1; i <= num_job_types; ++i)
        fprintf(outfile, "%8.3f", prob_distrib_job_type[i]);

    /* Initialize all machines in all stations to the idle state. */

    for (j = 1; j <= num_servers; ++j)
        num_machines_busy[j] = 0;

    /* Initialize simlib */

    //init_simlib();

    /* Set maxatr = max(maximum number of attributes per record, 4) */

    maxatr = 4;  /* NEVER SET maxatr TO BE SMALLER THAN 4. */

    /* Schedule the arrival of the first job. */

    event_schedule(expon(mean_interarrival, STREAM_INTERARRIVAL),
                   EVENT_ARRIVAL);

    /* Schedule the end of the simulation.  (This is needed for consistency of
       units.) */

    event_schedule(60 * length_simulation, EVENT_END_SIMULATION);

    /* Run the simulation until it terminates after an end-simulation event
       (type EVENT_END_SIMULATION) occurs. */

    do {

        /* Determine the next event. */

        timing();

        /* Invoke the appropriate event function. */

        switch (next_event_type) {
            case EVENT_ARRIVAL:
                arrive(1);
                break;
            case EVENT_DEPARTURE:
                depart();
                break;
            case EVENT_END_SIMULATION:
                report();
                break;
        }

    /* If the event just executed was not the end-simulation event (type
       EVENT_END_SIMULATION), continue simulating.  Otherwise, end the
       simulation. */

    } while (next_event_type != EVENT_END_SIMULATION);

    //fclose(infile);
    //fclose(outfile);

    return 0;
}


void arrive(int new_job)  /* Function to serve as both an arrival event of a job
                             to the system, as well as the non-event of a job's
                             arriving to a subsequent station along its
                             route. */
{
    int server;

    /* If this is a new arrival to the system, generate the time of the next
       arrival and determine the job type and task number of the arriving
       job. */


    if (new_job == 1) {

        event_schedule(sim_time + expon(mean_interarrival, STREAM_INTERARRIVAL),
                       EVENT_ARRIVAL);
        job_type = random_integer(prob_distrib_job_type, STREAM_JOB_TYPE);
        //task     = 1;
    }

    /* Determine the station from the route matrix. */

    server = job_type;

    /* Check to see whether all machines in this station are busy. */

    if (num_machines_busy[server] == num_vms[server]) {

        /* All machines in this station are busy, so place the arriving job at
           the end of the appropriate queue. Note that the following data are
           stored in the record for each job:
             1. Time of arrival to this station.
             2. Job type.
             3. Current task number. */

        transfer[1] = sim_time;
        transfer[2] = job_type;
        //transfer[3] = task;
        list_file(LAST, server);
    }

    else {

        /* A machine in this station is idle, so start service on the arriving
           job (which has a delay of zero). */

        sampst(0.0, server);                              /* For station. */
        sampst(0.0, num_servers + job_type);              /* For job type. */
        ++num_machines_busy[server];
        timest((float) num_machines_busy[server], server);

        /* Schedule a service completion.  Note defining attributes beyond the
           first two for the event record before invoking event_schedule. */

        transfer[3] = job_type;
        //transfer[4] = task;
        event_schedule(sim_time
                       + erlang(2, mean_service,
                                STREAM_SERVICE),
                       EVENT_DEPARTURE);
    }
}


void depart(void)  /* Event function for departure of a job from a particular
                      station. */
{
    int server, job_type_queue;
    //int task_queue;

    /* Determine the station from which the job is departing. */

    job_type = transfer[3];
    //task     = transfer[4];
    server  = job_type;

    /* Check to see whether the queue for this station is empty. */

    if (list_size[server] == 0) {

        /* The queue for this station is empty, so make a machine in this
           station idle. */

        --num_machines_busy[server];
        timest((float) num_machines_busy[server], server);
    }

    else {

        /* The queue is nonempty, so start service on first job in queue. */

        list_remove(FIRST, server);

        /* Tally this delay for this station. */

        sampst(sim_time - transfer[1], server);

        /* Tally this same delay for this job type. */

        job_type_queue = transfer[2];
        //task_queue     = transfer[3];
        sampst(sim_time - transfer[1], num_servers + job_type_queue);

        /* Schedule end of service for this job at this station.  Note defining
           attributes beyond the first two for the event record before invoking
           event_schedule. */

        transfer[3] = job_type_queue;
        //transfer[4] = task_queue;
        event_schedule(sim_time
                       + expon(mean_service,
                                STREAM_SERVICE),
                       EVENT_DEPARTURE);
}

    /* If the current departing job has one or more tasks yet to be done, send
       the job to the next station on its route. */

    //if (task < num_tasks[job_type]) {
    //    ++task;
    //    arrive(2);
    //}
}


void report(void)  /* Report generator function. */
{
    int   i;
    float overall_avg_job_tot_delay, avg_job_tot_delay, sum_probs;

    /* Compute the average total delay in queue for each job type and the
       overall average job total delay. */

    //fprintf(outfile, "\n\n\n\nServer type     Average total delay in queue");
    overall_avg_job_tot_delay = 0.0;
    sum_probs                 = 0.0;
    for (i = 1; i <= num_job_types; ++i) {
        avg_job_tot_delay = sampst(0.0, -(num_servers + i)) * 1.0;
        //fprintf(outfile, "\n\n%4d%27.3f", i, avg_job_tot_delay);
        overall_avg_job_tot_delay += (prob_distrib_job_type[i] - sum_probs)
                                     * avg_job_tot_delay;
        sum_probs = prob_distrib_job_type[i];
    }
    fprintf(outfile, "\n\nOverall average server total delay =%10.3f\n",
            overall_avg_job_tot_delay);
    fprintf(outfiles, "\n%.3f\n", overall_avg_job_tot_delay);

    /* Compute the average number in queue, the average utilization, and the
       average delay in queue for each station. */

    fprintf(outfile,
           "\n\n\n Edge      Average number      Average       Average delay");
    fprintf(outfile,
             "\nserver       in queue       utilization        in queue");
    for (j = 1; j <= num_servers; ++j)
        fprintf(outfile, "\n\n%4d%17.3f%17.3f%17.3f", j, filest(j),
                timest(0.0, -j) / num_vms[j], sampst(0.0, -j));
}

