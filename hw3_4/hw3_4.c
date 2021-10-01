#include "simlib.h"              /* Required for use of simlib.c. */


#define PI 3.14159265358979323846
#define STREAM_Z 3
#define STREAM_U 4

float normal_pdf(float epsilon);
float normal_cdf(float x);
float phi(float z);
float sample_mean(float z[]);
float sample_variance(float z[]);
float desired2actual_variance(float var);

float desired_mean = 10.0;
float desired_var = 16.0;
float desired_std = 4.0;
float a = 0.0;
float b = 20.0;
int sample_size = 20000;
float result[30000];

int main(){
    float sum = 0;
    float v_sum = 0;
    float sample_mean, sample_variance;
    float actual_var = desired2actual_variance(desired_var);
    float actual_std = sqrt(actual_var);

    // reference from: https://link.springer.com/content/pdf/10.1007/BF00143942.pdf
    for (int i = 0; i < sample_size; i ++){
        float x, z, phi_z, u;
        x = uniform(a, b, STREAM_Z);
        z = (x - desired_mean) / actual_std;
        phi_z = phi(z);
        u = lcgrand(STREAM_U);
        while(u > phi_z){
            x = uniform(a, b, STREAM_Z);
            z = (x - desired_mean) / actual_std;
            phi_z = phi(z);
            u = lcgrand(STREAM_U);
        }
        result[i] = x;
        sum += x;
    }
    sample_mean = sum/sample_size;
    for (int i = 0; i < sample_size; i ++){
        v_sum += (result[i] - sample_mean) * (result[i] - sample_mean);
    }
    sample_variance = v_sum/(sample_size-1);
    printf("actual variance for normal = %f\n", actual_var);
    printf("actual std for normal = %f\n", actual_std);
    printf("sample mean for 20000 samples = %f\n", sample_mean);
    printf("sample variance for 20000 samples = %f\n", sample_variance);
    printf("sample std for 20000 samples = %f\n", sqrt(sample_variance));
    
    FILE  *outfile;
    outfile = fopen("truncated_normal.txt", "w");
    for (int i = 0; i < sample_size; i ++){
        fprintf(outfile, "%f\n", result[i]);
    }

    return 0;
}

float phi(float z){
    return exp(-z*z/2);
}

float normal_pdf(float epsilon){
    return 1/(sqrt(2 * PI)) * exp(-0.5 * epsilon * epsilon);
}

float normal_cdf(float x){
    return 0.5 * (1 + erff(x / (sqrt(2.0))));
}

// calculate actual variance that would make the truncated normal variance desired
// approximation using bisection method
float desired2actual_variance(float var){
    float alpha, beta, phi_alpha, phi_beta, Z, temp, actual_var;
    float low = 1.0;
    float high = desired_var * 5.0;
    while ( high-low > 1e-5){
        actual_var = (high + low) / 2.0;
        float actual_std = sqrt(actual_var);
        alpha = (a - desired_mean) / actual_std;
        beta = (b - desired_mean) / actual_std;
        phi_alpha = normal_pdf(alpha);
        phi_beta = normal_pdf(beta);
        Z = normal_cdf(beta) - normal_cdf(alpha);
        // formula from wiki
        temp = actual_var * (1.0 + (alpha * phi_alpha - beta * phi_beta) / (Z) - ((phi_alpha - phi_beta) / Z)*((phi_alpha - phi_beta) / Z));
        if (abs(desired_var - temp) < 1e-5){
            return actual_var;
        }else if (desired_var > temp){
            low = actual_var;
        }else if (desired_var < temp){
            high = actual_var;
        }
    }
    return actual_var;
}