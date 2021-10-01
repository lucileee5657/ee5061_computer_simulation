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
#define MAX_NUM_STATIONS      5  /* Maximum number of stations. */
#define MAX_NUM_JOB_TYPES     3  /* Maximum number of job types. */

/* Declare non-simlib global variables. */

int   num_stations, i, j, num_machines[MAX_NUM_STATIONS + 1],
      num_machines_busy[MAX_NUM_STATIONS + 1], job_type, task;
float mean_interarrival, length_simulation, prob_repair, prob_inspect, arrival_rate,
      service[MAX_NUM_STATIONS + 1][MAX_NUM_STATIONS + 1];
FILE  *infile, *outfile, *logfile;

/* Declare non-simlib functions. */

void  arrive(int new_job);
void  depart(void);
void  report(void);


int main()  /* Main function. */
{
    /* Open input and output files. */

    infile  = fopen("inspect_repair.in",  "r");
    outfile = fopen("inspect_repair.out", "w");
    logfile = fopen("inspect_repair.log", "w");
    /* Read input parameters. */

    fscanf(infile, "%d %f %f", &num_stations, 
           &arrival_rate, &length_simulation);
    for (j = 1; j <= num_stations; ++j)
        fscanf(infile, "%d", &num_machines[j]);
    for (i = 1; i <= num_stations; ++i) {
        fscanf(infile, "%f %f ", &service[i][1], &service[i][2]);
    }
    fscanf(infile, "%f %f", &prob_repair, &prob_inspect);
    
    mean_interarrival = 1 / arrival_rate;
    /* Write report heading and input parameters. */

    fprintf(outfile, "Bus inspection and repair model\n\n");
    fprintf(outfile, "Number of work stations%21d\n\n", num_stations);
    fprintf(outfile, "Number of machines in each station     ");
    for (j = 1; j <= num_stations; ++j)
        fprintf(outfile, "%5d", num_machines[j]);
    fprintf(outfile, "\n\nDistribution function of inspection and reapir times  ");
    for (i = 1; i <= num_stations; ++i)
        fprintf(outfile, "%8.3f %8.3f", service[i][1], service[i][2]);
    fprintf(outfile, "\n\nArrival rate of bus%20.2f per hours\n\n",
            arrival_rate);
    fprintf(outfile, "Mean interarrival time of bus%14.2f hours\n\n",
            mean_interarrival);
    
    fprintf(outfile, "Length of the simulation%20.1f hours \n\n",
            length_simulation);
    fprintf(outfile, "Prob to repair %25.1f\n\n", prob_repair);
    fprintf(outfile, "prob to re-inspect %21.1f\n\n", prob_inspect);
    /* Initialize all machines in all stations to the idle state. */

    for (j = 1; j <= num_stations; ++j)
        num_machines_busy[j] = 0;

    /* Initialize simlib */

    init_simlib();

    /* Set maxatr = max(maximum number of attributes per record, 4) */

    maxatr = 4;  /* NEVER SET maxatr TO BE SMALLER THAN 4. */

    /* Schedule the arrival of the first job. */

    event_schedule(expon(mean_interarrival, STREAM_INTERARRIVAL),
                   EVENT_ARRIVAL);

    /* Schedule the end of the simulation.  (This is needed for consistency of
       units.) */

    event_schedule(length_simulation, EVENT_END_SIMULATION);

    /* Run the simulation until it terminates after an end-simulation event
       (type EVENT_END_SIMULATION) occurs. */

    do {

        /* Determine the next event. */

        timing();

        /* Invoke the appropriate event function. */

        switch (next_event_type) {
            case EVENT_ARRIVAL:
                //fprintf(outfile, "new task @ %f\n", sim_time);
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

    fclose(infile);
    fclose(outfile);
    fclose(logfile);

    return 0;
}


void arrive(int new_job)  /* Function to serve as both an arrival event of a job
                             to the system, as well as the non-event of a job's
                             arriving to a subsequent station along its
                             route. */
{
    int station;

    /* If this is a new arrival to the system, generate the time of the next
       arrival and determine the job type and task number of the arriving
       job. */
    fprintf(logfile, "new task @ %f\n", sim_time);

    if (new_job == 1) {
        
        event_schedule(sim_time + expon(mean_interarrival, STREAM_INTERARRIVAL),
                       EVENT_ARRIVAL);
        task     = 1;
    }

    /* Determine the station from the route matrix. */

    station = task;

    /* Check to see whether all machines in this station are busy. */

    if (num_machines_busy[station] == num_machines[station]) {

        /* All machines in this station are busy, so place the arriving job at
           the end of the appropriate queue. Note that the following data are
           stored in the record for each job:
             1. Time of arrival to this station.
             2. Job type.
             3. Current task number. */

        transfer[1] = sim_time;
        //transfer[2] = job_type;
        transfer[3] = task;
        list_file(LAST, station);
    }

    else {

        /* A machine in this station is idle, so start service on the arriving
           job (which has a delay of zero). */

        sampst(0.0, station);                              /* For station. */
        //sampst(0.0, num_stations + job_type);              /* For job type. */
        ++num_machines_busy[station];
        timest((float) num_machines_busy[station], station);

        /* Schedule a service completion.  Note defining attributes beyond the
           first two for the event record before invoking event_schedule. */

        //transfer[3] = job_type;
        transfer[4] = task;
        event_schedule(sim_time
                       + uniform(service[task][1], service[task][2],
                                STREAM_SERVICE),
                       EVENT_DEPARTURE);
    }
}


void depart(void)  /* Event function for departure of a job from a particular
                      station. */
{
    int station, job_type_queue, task_queue;

    /* Determine the station from which the job is departing. */

    //job_type = transfer[3];
    task     = transfer[4];
    station  = task;

    /* Check to see whether the queue for this station is empty. */

    if (list_size[station] == 0) {

        /* The queue for this station is empty, so make a machine in this
           station idle. */

        --num_machines_busy[station];
        timest((float) num_machines_busy[station], station);
    }

    else {

        /* The queue is nonempty, so start service on first job in queue. */

        list_remove(FIRST, station);

        /* Tally this delay for this station. */

        sampst(sim_time - transfer[1], station);

        /* Tally this same delay for this job type. */

        //job_type_queue = transfer[2];
        task_queue     = transfer[3];
        //sampst(sim_time - transfer[1], num_stations);

        /* Schedule end of service for this job at this station.  Note defining
           attributes beyond the first two for the event record before invoking
           event_schedule. */

        //transfer[3] = job_type_queue;
        transfer[4] = task_queue;
        event_schedule(sim_time
                       + uniform(service[task_queue][1], service[task_queue][2],
                                STREAM_SERVICE),
                       EVENT_DEPARTURE);
        
    }

    /* If the current departing job has one or more tasks yet to be done, send
       the job to the next station on its route. */
    fprintf(logfile, "depart @ %f from %d station\n", sim_time, task);
    
    if (task == 1) {
        if (lcgrand(STREAM_JOB_TYPE) <= prob_repair){
            fprintf(logfile, "need repair...\n");
            ++task;
            arrive(2);
        } 
    }
    else if (task == 2) {
        if (lcgrand(STREAM_JOB_TYPE) <= prob_inspect){
            fprintf(logfile, "need re-inspect...\n");
            --task;
            arrive(2);
        } 
    }
    
    
}


void report(void)  /* Report generator function. */
{
    int   i;
    float overall_avg_job_tot_delay, avg_job_tot_delay, sum_probs;

    /* Compute the average total delay in queue for each job type and the
       overall average job total delay. */


    /* Compute the average number in queue, the average utilization, and the
       average delay in queue for each station. */

    fprintf(outfile,
           "\n\n\n Work      Average number      Average       Average delay");
    fprintf(outfile,
             "\nstation       in queue       utilization        in queue");
    
    fprintf(outfile, "\n\nIspection %12.3f%17.3f%17.3f", filest(1),
                timest(0.0, -1) / num_machines[1], sampst(0.0, -1));
    fprintf(outfile, "\n\n   Repair %12.3f%17.3f%17.3f", filest(2),
                timest(0.0, -2) / num_machines[2], sampst(0.0, -2));
    /*
    for (j = 1; j <= num_stations; ++j)
        fprintf(outfile, "\n\n%4d%17.3f%17.3f%17.3f", j, filest(j),
                timest(0.0, -j) / num_machines[j], sampst(0.0, -j));
    */
}

