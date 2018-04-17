/* 
Lucas Tata / Salvatore Amico
High Performance Computing on GPUs
Final Project
Collaborative Filtering
GPU Implementation
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include "../helper/util.h"
#include "../helper/wtime.h"

#define INPUT_SIZE 16384 //how many lines to read from the dataset
#define SPARSE_SIZE 8192 //size of sparse matrix is sparse_size * sparse_size
#define USER_SIZE 2048
#define ARTIST_SIZE 8192
#define LINE_SIZE 1024
#define RAND_RANGE 100 //sets the random number generation range
#define NUM_RECOMMENDATIONS 5 //number of recommendations to generate for the user
#define NUM_FEATURES 10 //number of features to generate for each user in the algorithm
#define ITERATIONS 1 //how many iterations you want to run the algorithm with
#define USER_ID 1 //indicates which user you want to generate recommendations for
#define SHARED_SIZE 100
#define SPLIT 25

int *dataMatrix, *X, *Y, *X_T, *Y_T; //our output sparse matrix (users by artists, data is the play count) 
char **artists;
char **users;
char **artistNames;
int endOfArtistIndex = 0; //keep tabs on how many artists are currently in there
int endOfUserIndex = 0; //keep tabs on how many users are currently in there

__global__ void gpu_als(int *x, int *user_row, int *user_pref, int * conf_I, int *conf, int num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < num)
    {
        for(int k = 0; k < NUM_FEATURES; k++)
        {
            user_row[tid * NUM_FEATURES + k] = x[tid*NUM_FEATURES + k];
            if(user_row[tid * NUM_FEATURES + k] != 0)
            {
                user_pref[tid*NUM_FEATURES + k] = 1;
            }
            else
            {
                user_pref[tid*NUM_FEATURES + k] = user_row[tid*NUM_FEATURES + k];
            }
            conf_I[tid*NUM_FEATURES * k * NUM_FEATURES + k] = user_row[tid * NUM_FEATURES + k];
            conf[tid*NUM_FEATURES * k * NUM_FEATURES + k] = user_row[tid * NUM_FEATURES + k] + 1;
        }
        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void gpu_mat_mat_multiply(int *a, int *b, int *c, int num_rows, int num_cols, int num_rows2)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < num_rows*num_rows2)
    {
        int res = 0;
        int row = tid / num_rows;
        int col = tid % num_rows;
        for(int j = 0; j < num_cols; j++)
        {
            res += a[row * num_cols + j] * b[col + num_rows2 * j];
        }
        
        c[tid] = res;
        tid+= blockDim.x * gridDim.x;
    }
}

__global__ void gpu_mat_mat_multiply_shared(int *a, int *b, int *c, int num_rows, int num_cols, int num_rows2)
{
    __shared__ int sres[NUM_FEATURES];
    int bid = blockIdx.x;
    while(bid < num_rows*num_rows2)
    {
        int row = bid / num_cols;
        int col = bid % num_cols;
        int tid = threadIdx.x;
        while(tid < num_cols)
        {
            sres[tid] = a[row * num_cols + tid] * b[col + num_rows2 * tid];
            tid += blockDim.x;
        }
        __syncthreads();
        for(int i = num_cols/2 ; i > 0; i /= 2)
        {
            if(threadIdx.x < i)
            {
                int temp = sres[threadIdx.x] + sres[threadIdx.x + i];
                sres[threadIdx.x] = temp;
            }
            __syncthreads();
        }
        if(threadIdx.x == 0)
        c[bid] = sres[threadIdx.x];
        bid += gridDim.x;
        __syncthreads();
    }
}

__global__ void gpu_mat_mat_multiply_atomic(int *a, int *b, int *c, int num_rows, int num_cols, int num_rows2)
{
    int res;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    while(bid * SPLIT < num_rows*num_cols)
    {
        int row = bid / num_cols + tid / SPLIT;
        int col = bid % num_cols;
        res = a[row * num_cols + tid%SPLIT] * b[col + num_rows2 * tid%SPLIT];
        __syncthreads();
        atomicAdd(&c[bid + tid/SPLIT], res);
        bid += gridDim.x;
        __syncthreads();
    }
}


__global__ void
gpu_mat_vec_multiply_shared(int *mat, int *vec, int *res, int num_rows, int num_cols)
{
    __shared__ int svec[256];
    __shared__ int sres[256];
    svec[threadIdx.x] = vec[threadIdx.x];
	int bid = blockIdx.x;
    __syncthreads();
	while (bid < num_rows)
	{
        sres[threadIdx.x] = mat[bid * num_cols + threadIdx.x] * svec[threadIdx.x];
        __syncthreads();
        for(int i = blockDim.x/2 ; i > 0; i /= 2)
        {
            if(threadIdx.x < i)
            {
                int temp = sres[threadIdx.x] + sres[threadIdx.x + i];
                sres[threadIdx.x] = temp;
            }
            __syncthreads();
        }
        if(threadIdx.x == 0)
        res[bid] = sres[threadIdx.x];
		bid += 128;
        __syncthreads();
	}
}


__global__ void gpu_matrix_transpose(int *mat, int *res, int num_rows, int num_cols)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < num_rows*num_cols)
    {
        int in_row = tid / num_cols;
        int in_col = tid % num_cols;
        int out_row = in_col;
        int out_col = in_row;
        res [out_row * num_cols + out_col] = mat [in_row * num_cols + in_col];
        tid += blockDim.x * gridDim.x;

    }
}

__global__ void gpu_matrix_alpha(int *mat, int *res, float alpha_val, int num_rows, int num_cols)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < num_rows*num_cols)
    {
        int in_row = tid / num_cols;
        int in_col = tid % num_cols;
        int out_row = in_col;
        int out_col = in_row;
        res [out_row * num_cols + out_col] = mat [in_row * num_cols + in_col];
        tid += blockDim.x * gridDim.x;

    }
}

__global__ void gpu_matrix_addition(int *mat1, int *mat2, int *res, int num_rows, int num_cols)
{
    int tid = threadIdx.x;
    while(tid < num_rows*num_cols)
    {
        int row = tid / num_cols;
        int col = tid % num_cols;
        res [row * num_cols + col] += mat1[row * num_cols + col] + mat2[row * num_cols + col];
        tid += blockDim.x * gridDim.x;

    }
}

__global__ void gpu_mat_div(int *mat1, int *mat2, int *res, int num_rows, int num_cols)
{
    int tid = threadIdx.x;
    while(tid < num_rows*num_cols)
    {
        int row = tid / num_cols;
        int col = tid % num_cols;
        res [row * num_cols + col] = mat1[row * num_cols + col] / mat2[row * num_cols + col];
        tid += blockDim.x * gridDim.x;

    }
}


int checkIfArtistExistsInData(char * artist)
{
    int i;
    for(i = 0; i < ARTIST_SIZE; i++)
    {
        if(strcmp(artist, artists[i]) == 0)
        {
            return i;
        }
    }
    return -1;
}

int checkIfUserExistsInData(char * user)
{
    int i;
    for(i = 0; i < USER_SIZE; i++)
    {
        if(strcmp(user, users[i]) == 0)
        {
            return i;
        }
    }
    return -1;
}

void mat_mat_multiply(int *mat1, int *mat2, int *res, int num_rows1, int num_cols, int num_rows2)
{
    for(int k = 0; k < num_rows1; k ++)
    {
        for(int i = 0; i < num_rows2; i ++)
        {
            int temp_res = 0;
            for (int j = 0; j < num_cols; j ++)
            {

                temp_res += mat1[i * num_cols + j] * mat2[k + num_rows2 * j];
            }

            res[k+i*num_rows1] = temp_res;
        }
    }
}

void mat_vec_multiply(int *mat, int *vec, int *res, int num_rows, int num_cols)
{
	for(int i = 0; i < num_rows; i ++)
	{
		int temp_res = 0;
		for (int j = 0; j < num_cols; j ++)
		{
			temp_res += mat[i * num_cols + j] * vec[j];
		}

		res[i] = temp_res;
	}
}


void recommend(int user_id, int num_items, int * answer)
{
    int *user_recs, *rec_vector, *X_rec;
    user_recs = (int *)malloc(sizeof(int) * endOfArtistIndex);
    rec_vector = (int *)malloc(sizeof(int) * endOfArtistIndex);
    X_rec = (int *)malloc(sizeof(int) * NUM_FEATURES);
    int maxVal = 0, index = 0, no = 0;
    for(int i = 0; i < endOfArtistIndex; i++)
    {
        
        user_recs[i] = dataMatrix[user_id*SPARSE_SIZE + i];
        if(user_recs[i] == 0)
        {
            user_recs[i] = 1;
        }
        else
        {
            user_recs[i] = 0;
        }
    }

    for(int i = 0; i < NUM_FEATURES; i++)
    {
        X_rec[i] = X[user_id * NUM_FEATURES + i];
    }

	//*********GPU*********//

    

    mat_vec_multiply(Y_T, X_rec, rec_vector, NUM_FEATURES, endOfArtistIndex); 


    /*int *mat_d, *vec_d, *res_vec_d;
    cudaMalloc((void **)&mat_d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);
    cudaMalloc((void **)&vec_d, sizeof(int) * endOfArtistIndex);
    cudaMalloc((void **)&res_vec_d, sizeof(int) * endOfArtistIndex);

    cudaMemcpy(mat_d, Y_T, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
    cudaMemcpy(vec_d, X_rec, sizeof(int) * endOfArtistIndex, cudaMemcpyHostToDevice);

    gpu_mat_vec_multiply_shared<<<256, 256>>>(mat_d, vec_d, res_vec_d, NUM_FEATURES, endOfArtistIndex);

    cudaMemcpy(rec_vector, res_vec_d, sizeof(int) * endOfArtistIndex, cudaMemcpyDeviceToHost);*/
	//*********GPU*********//

    for(int i = 0; i < num_items; i++)
    {
        
        maxVal = INT_MIN, index = 0;
        for(int j = 0; j < endOfArtistIndex; j++)
        {
            no = 0;
            if(rec_vector[j] > maxVal)
            {
                for(int k = 0; k < i; k++)
                {
                    if(j == answer[k])
                    {
                        no = 1;
                    }
                }
                if(no == 0)
                {
                    maxVal = rec_vector[j];
                    index = j;
                }
            }
        }
        answer[i] = index;
    }
}

int implicit_als(int alpha_val, int iterations, double lambda_val, int features)
{

    size_t available, total;
    cudaMemGetInfo(&available, &total);
    //printf("%u %u\n", available, total);
    double time_beg = wtime();

    
    //GPU alpha mult//
    for(int i = 0; i < endOfArtistIndex; i++)
    {
        for(int j = 0; j < endOfUserIndex; j++)
        {
            dataMatrix[i * SPARSE_SIZE + j] = dataMatrix[i * SPARSE_SIZE + j] * alpha_val;
        }
    }
    //NEWGPU//
    
    int *X_P, *Y_P; 

    X = (int *)malloc(sizeof(int) * endOfUserIndex * NUM_FEATURES);
    Y = (int *)malloc(sizeof(int) * endOfArtistIndex * NUM_FEATURES);
    X_T = (int *)malloc(sizeof(int) * endOfUserIndex * NUM_FEATURES);
    Y_T = (int *)malloc(sizeof(int) * endOfArtistIndex * NUM_FEATURES);
    
    
    X_P = (int *)malloc(sizeof(int) * endOfUserIndex * endOfUserIndex);
    Y_P = (int *)malloc(sizeof(int) * endOfArtistIndex * endOfArtistIndex);
    
    //GPU random//
    for(int i = 0; i < endOfUserIndex; i++)
    {
        for(int j = 0; j < features; j++)
        {
            X[i * features + j] = rand() % RAND_RANGE;
        }
    }
    //NEWGPU//

    //GPU transpose//
    /*for(int i = 0; i < endOfUserIndex; i++)
    {
        for(int j = 0; j < features; j++)
        {
            X_T[j * endOfUserIndex + i] = X[i*features + j];
        }
    }*/

    int *m1, *t1;
    H_ERR(cudaMalloc((void **)&m1, sizeof(int) * endOfUserIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&t1, sizeof(int) * endOfUserIndex * NUM_FEATURES));

    H_ERR(cudaMemcpy(m1, X, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));

    gpu_matrix_transpose<<<256, 256>>>(m1, t1, endOfUserIndex, NUM_FEATURES);

    H_ERR(cudaMemcpy(X_T, t1, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyDeviceToHost));

    H_ERR(cudaFree(m1));
    H_ERR(cudaFree(t1));

    

    //NEWGPU//



    //GPU random//
    for(int i = 0; i < endOfArtistIndex; i++)
    {
        for(int j = 0; j < features; j++)
        {
            Y[i * features + j] = rand() % RAND_RANGE;
        }
    }

    //GPU transpose
    /*for(int i = 0; i < endOfArtistIndex; i++)
    {
        for(int j = 0; j < features; j++)
        {
            Y_T[j * endOfArtistIndex + i] = Y[i * features + j];
        }
    }*/
    

    int *m2, *t2;
    H_ERR(cudaMalloc((void **)&m2, sizeof(int) * endOfUserIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&t2, sizeof(int) * endOfUserIndex * NUM_FEATURES));

    H_ERR(cudaMemcpy(m2, Y, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));

    gpu_matrix_transpose<<<256, 256>>>(m2, t2, endOfUserIndex, NUM_FEATURES);

    H_ERR(cudaMemcpy(Y_T, t2, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyDeviceToHost));

    H_ERR(cudaFree(m2));
    H_ERR(cudaFree(t2));

    double elapsed_time = wtime();
    elapsed_time -= time_beg;
    //printf("setup elapsed time is: %f\n", elapsed_time);

    time_beg = wtime();


    int *X_I, *Y_I, *I, *I1, *user_row, *artist_row, *user_pref, *artist_pref, *user_confidence, *artist_confidence, *user_confidence_I, *artist_confidence_I; 

    X_I = (int *)malloc(sizeof(int) * endOfUserIndex * endOfUserIndex);
    Y_I = (int *)malloc(sizeof(int) * endOfArtistIndex * endOfArtistIndex);
    I = (int *)malloc(sizeof(int) * features * features);
    I1 = (int *)malloc(sizeof(int) * endOfArtistIndex * endOfArtistIndex);
    user_row = (int *)malloc(sizeof(int) * endOfArtistIndex * features);
    artist_row = (int *)malloc(sizeof(int) * endOfUserIndex * features);
    user_pref = (int *)malloc(sizeof(int) * endOfArtistIndex * features);
    artist_pref = (int *)malloc(sizeof(int) * endOfUserIndex * features);
    user_confidence = (int *)malloc(sizeof(int) * endOfUserIndex * features * features);
    artist_confidence = (int *)malloc(sizeof(int) * endOfArtistIndex * features * features);
    user_confidence_I = (int *)malloc(sizeof(int) * endOfUserIndex * features * features);
    artist_confidence_I = (int *)malloc(sizeof(int) * endOfArtistIndex * features * features);

    int *X_temp, *Y_temp, *Y_result_y, *Y_result_pu, *Y_temp_2, *X_result_x, *X_result_pi, *X_temp_2;
    X_temp = (int *)malloc(sizeof(int) * endOfUserIndex * features);
    X_temp_2 = (int *)malloc(sizeof(int) * endOfUserIndex * features);
    X_result_x = (int *)malloc(sizeof(int) * endOfUserIndex * endOfUserIndex);
    X_result_pi = (int *)malloc(sizeof(int) * endOfUserIndex * endOfUserIndex);
    Y_temp = (int *)malloc(sizeof(int) * endOfArtistIndex * features);
    Y_temp_2 = (int *)malloc(sizeof(int) * endOfArtistIndex * features);
    Y_result_y = (int *)malloc(sizeof(int) * endOfArtistIndex * endOfArtistIndex);
    Y_result_pu = (int *)malloc(sizeof(int) * endOfArtistIndex * endOfArtistIndex);


    //GPU???///
    for(int i = 0; i < endOfUserIndex; i++)
    {
        X_I[i * endOfUserIndex + i] = 1;
    }

    for(int i = 0; i < endOfArtistIndex; i++)
    {
        Y_I[i * endOfArtistIndex + i] = 1;
    }

    for(int i = 0; i < features; i++)
    {
        I[i * features + i] = 1;
    }
    for(int i = 0; i < endOfArtistIndex; i++)
    {
        I1[i * endOfArtistIndex + i] = lambda_val;
    }

	//*********GPU*********//

    
    //mat_mat_multiply(X, X_T, X_P, endOfUserIndex, features, endOfUserIndex);
    int *mat1_d, *mat2_d, *res_d;
    H_ERR(cudaMalloc((void **)&mat1_d, sizeof(int) * endOfUserIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&mat2_d, sizeof(int) * endOfUserIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&res_d, sizeof(int) * endOfUserIndex * endOfUserIndex));

    

    H_ERR(cudaMemcpy(mat1_d, X, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(mat2_d, X_T, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));

    

    gpu_mat_mat_multiply_atomic<<<256, 256>>>(mat1_d, mat2_d, res_d, endOfUserIndex, NUM_FEATURES, endOfUserIndex);

    H_ERR(cudaMemcpy(X_P, res_d, sizeof(int) * endOfUserIndex * endOfUserIndex, cudaMemcpyDeviceToHost));

    

    H_ERR(cudaFree(mat1_d));
    H_ERR(cudaFree(mat2_d));
    H_ERR(cudaFree(res_d));

    

    
    //mat_mat_multiply(Y, Y_T, Y_P, endOfArtistIndex, features, endOfArtistIndex);//
    int *mat1_2d, *mat2_2d, *res_2d;

    H_ERR(cudaMalloc((void **)&mat1_2d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&mat2_2d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&res_2d, sizeof(int) * endOfArtistIndex * endOfArtistIndex));

    H_ERR(cudaMemcpy(mat1_2d, Y, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(mat2_2d, Y_T, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));

    gpu_mat_mat_multiply_atomic<<<256, 256>>>(mat1_2d, mat2_2d, res_2d, endOfArtistIndex, NUM_FEATURES, endOfArtistIndex);

    cudaMemGetInfo(&available, &total);
    //printf("%u %u\n", available, total);

    H_ERR(cudaMemcpy(Y_P, res_2d, sizeof(int) * endOfArtistIndex * endOfArtistIndex, cudaMemcpyDeviceToHost));

    
    

    H_ERR(cudaFree(mat1_2d));
    H_ERR(cudaFree(mat2_2d));
    H_ERR(cudaFree(res_2d));

	//*********GPU*********//

    elapsed_time = wtime();
    elapsed_time -= time_beg;
    //printf("part 1 elapsed time is: %f\n", elapsed_time);

    time_beg = wtime();

    int *x_d, *user_row_d, *user_pref_d, *conf_I_d, *conf_d;

    H_ERR(cudaMalloc((void **)&x_d, sizeof(int) * endOfUserIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&user_row_d, sizeof(int) * endOfArtistIndex * features));
    H_ERR(cudaMalloc((void **)&user_pref_d, sizeof(int) * endOfArtistIndex * features));
    H_ERR(cudaMalloc((void **)&conf_I_d, sizeof(int) * endOfUserIndex * NUM_FEATURES * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&conf_d, sizeof(int) * endOfUserIndex * NUM_FEATURES * NUM_FEATURES));


    H_ERR(cudaMemcpy(x_d, X, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));


    gpu_als<<<256, 256>>>(x_d, user_row_d, user_pref_d, conf_I_d, conf_d, endOfUserIndex);


    
    H_ERR(cudaMemcpy(user_row, user_row_d, sizeof(int) * endOfArtistIndex * features, cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(user_pref, user_pref_d, sizeof(int) * endOfArtistIndex * features, cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(user_confidence_I, conf_I_d, sizeof(int) * endOfUserIndex * NUM_FEATURES * NUM_FEATURES, cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(user_confidence, conf_d, sizeof(int) * endOfUserIndex * NUM_FEATURES * NUM_FEATURES, cudaMemcpyDeviceToHost));

    H_ERR(cudaFree(x_d));
    H_ERR(cudaFree(user_row_d));
    H_ERR(cudaFree(user_pref_d));
    H_ERR(cudaFree(conf_I_d));
    H_ERR(cudaFree(conf_d));


    int *mat1_3d, *mat2_3d, *res_3d;
    H_ERR(cudaMalloc((void **)&mat1_3d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&mat2_3d, sizeof(int) * NUM_FEATURES * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&res_3d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));

    int *mat1_4d, *mat2_4d, *res_4d;
    H_ERR(cudaMalloc((void **)&mat1_4d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&mat2_4d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&res_4d, sizeof(int) * endOfArtistIndex * endOfArtistIndex));

    int *mat1_add, *mat2_add, *res_add;
    H_ERR(cudaMalloc((void **)&mat1_add, sizeof(int) * endOfArtistIndex * endOfArtistIndex));
    H_ERR(cudaMalloc((void **)&mat2_add, sizeof(int) * endOfArtistIndex * endOfArtistIndex));
    H_ERR(cudaMalloc((void **)&res_add, sizeof(int) * endOfArtistIndex * endOfArtistIndex));


    int *mat1_5d, *mat2_5d, *res_5d;
    H_ERR(cudaMalloc((void **)&mat1_5d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&mat2_5d, sizeof(int) * NUM_FEATURES * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&res_5d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));

    int *mat1_6d, *mat2_6d, *res_6d;
    H_ERR(cudaMalloc((void **)&mat1_6d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&mat2_6d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&res_6d, sizeof(int) * endOfArtistIndex * endOfArtistIndex));

    H_ERR(cudaMemcpy(mat1_3d, Y_T, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(mat2_4d, Y, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(mat1_5d, Y_T, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(mat2_6d, Y, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));

    H_ERR(cudaMemcpy(mat1_add, Y_P, sizeof(int) * endOfArtistIndex * endOfArtistIndex, cudaMemcpyHostToDevice));


    for(int i = 0; i < iterations; i++)
    {
        for(int j = 0; j < endOfUserIndex; j++)
        {
            /*for(int k = 0; k < features; k++)
            {
                user_row[k] = X[j*features + k];
                if(user_row[k] != 0)
                {
                    user_pref[k] = 1;
                }
                else
                {
                    user_pref[k] = user_row[k];
                }
            }

            for(int k = 0; k < features; k++)
            {
                user_confidence_I[k * features + k] = user_row[k];
                user_confidence[k * features + k] = user_row[k] + 1;
            }*/




            elapsed_time = wtime();
            elapsed_time -= time_beg;
            //printf("part 2 elapsed time is: %f\n", elapsed_time);

            time_beg = wtime();
            //*********GPU*********//
             //mat_mat_multiply(Y_T, user_confidence_I, Y_temp, endOfArtistIndex, features, features);//
            
            H_ERR(cudaMemcpy(mat2_3d, user_confidence_I + j * NUM_FEATURES, sizeof(int) * NUM_FEATURES * NUM_FEATURES, cudaMemcpyHostToDevice));

            gpu_mat_mat_multiply_atomic<<<256, 256>>>(mat1_3d, mat2_3d, res_3d, endOfArtistIndex, NUM_FEATURES, NUM_FEATURES);

            H_ERR(cudaMemcpy(Y_temp, res_3d, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyDeviceToHost));

            

            
           
            //mat_mat_multiply(Y_temp, Y, Y_result_y, endOfArtistIndex, features, endOfArtistIndex);//

            elapsed_time = wtime();
            elapsed_time -= time_beg;
            //printf("part 3 elapsed time is: %f\n", elapsed_time);

            time_beg = wtime();

            
            H_ERR(cudaMemcpy(mat1_4d, Y_temp, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
            
            gpu_mat_mat_multiply_atomic<<<256, 256>>>(mat1_4d, mat2_4d, res_4d, endOfArtistIndex, NUM_FEATURES, endOfArtistIndex);

            H_ERR(cudaMemcpy(Y_result_y, res_4d, sizeof(int) * endOfArtistIndex * endOfArtistIndex, cudaMemcpyDeviceToHost));
            
			//*********GPU*********//
            /*for(int j = 0; j < endOfArtistIndex; j++)
            {
                for(int k = 0; k < endOfArtistIndex; k++)
                {
                    Y_result_y[j*endOfArtistIndex + k] += Y_P[j*endOfArtistIndex + k] + I1[j*endOfArtistIndex + k];
                }
            }*/

            
            H_ERR(cudaMemcpy(mat2_add, I1, sizeof(int) * endOfArtistIndex * endOfArtistIndex, cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(res_add, Y_result_y, sizeof(int) * endOfArtistIndex * endOfArtistIndex, cudaMemcpyHostToDevice));

            gpu_matrix_addition<<<256, 256>>>(mat1_add, mat2_add, res_add, endOfArtistIndex, endOfArtistIndex);

            H_ERR(cudaMemcpy(Y_result_pu, res_add, sizeof(int) * endOfArtistIndex * endOfArtistIndex, cudaMemcpyDeviceToHost));

            

			//*********GPU*********//
            elapsed_time = wtime();
            elapsed_time -= time_beg;
            //printf("part 4 elapsed time is: %f\n", elapsed_time);

            time_beg = wtime();
            //mat_mat_multiply(Y_T, user_confidence, Y_temp_2, endOfArtistIndex, features, features);//
            
            H_ERR(cudaMemcpy(mat2_5d, user_confidence + j * NUM_FEATURES, sizeof(int) * NUM_FEATURES * NUM_FEATURES, cudaMemcpyHostToDevice));

            gpu_mat_mat_multiply_atomic<<<256, 256>>>(mat1_5d, mat2_5d, res_5d, endOfArtistIndex, NUM_FEATURES, NUM_FEATURES);

            H_ERR(cudaMemcpy(Y_temp_2, res_5d, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyDeviceToHost));

            

            //mat_mat_multiply(Y_temp_2, Y, Y_result_pu, endOfArtistIndex, features, endOfArtistIndex);//
            elapsed_time = wtime();
            elapsed_time -= time_beg;
            //printf("part 6 elapsed time is: %f\n", elapsed_time);

            time_beg = wtime();

            
            H_ERR(cudaMemcpy(mat1_6d, Y_temp_2, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
            
            gpu_mat_mat_multiply_atomic<<<256, 256>>>(mat1_6d, mat2_6d, res_6d, endOfArtistIndex, NUM_FEATURES, endOfArtistIndex);

            H_ERR(cudaMemcpy(Y_result_pu, res_6d, sizeof(int) * endOfArtistIndex * endOfArtistIndex, cudaMemcpyDeviceToHost));

            

			//*********GPU*********//
            /*for(int k = 0; k < features; k++)
            {
                X[i*features + k] = Y_result_y[i*features + k] / Y_result_pu[i*features + k];
            }*/

            

        }

        H_ERR(cudaFree(mat1_4d));
        H_ERR(cudaFree(mat2_4d));
        H_ERR(cudaFree(res_4d));

        H_ERR(cudaFree(mat1_3d));
        H_ERR(cudaFree(mat2_3d));
        H_ERR(cudaFree(res_3d));

        H_ERR(cudaFree(mat1_add));
        H_ERR(cudaFree(mat2_add));
        H_ERR(cudaFree(res_add));

        H_ERR(cudaFree(mat1_5d));
        H_ERR(cudaFree(mat2_5d));
        H_ERR(cudaFree(res_5d));

        H_ERR(cudaFree(mat1_6d));
        H_ERR(cudaFree(mat2_6d));
        H_ERR(cudaFree(res_6d));

        int *y_d, *artist_row_d, *artist_pref_d, *art_conf_I_d, *art_conf_d;

    H_ERR(cudaMalloc((void **)&y_d, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&artist_row_d, sizeof(int) * endOfUserIndex * features));
    H_ERR(cudaMalloc((void **)&artist_pref_d, sizeof(int) * endOfUserIndex * features));
    H_ERR(cudaMalloc((void **)&art_conf_I_d, sizeof(int) * endOfArtistIndex * NUM_FEATURES * NUM_FEATURES));
    H_ERR(cudaMalloc((void **)&art_conf_d, sizeof(int) * endOfArtistIndex * NUM_FEATURES * NUM_FEATURES));


    H_ERR(cudaMemcpy(y_d, Y, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));


    gpu_als<<<256, 256>>>(y_d, artist_row_d, artist_pref_d, art_conf_I_d, art_conf_d, endOfArtistIndex);


    H_ERR(cudaMemcpy(artist_row, artist_row_d, sizeof(int) * endOfUserIndex * features, cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(artist_pref, artist_pref_d, sizeof(int) * endOfUserIndex * features, cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(artist_confidence_I, art_conf_I_d, sizeof(int) * endOfArtistIndex * NUM_FEATURES * NUM_FEATURES, cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(artist_confidence, art_conf_d, sizeof(int) * endOfArtistIndex * NUM_FEATURES * NUM_FEATURES, cudaMemcpyDeviceToHost));


    H_ERR(cudaFree(y_d));
    H_ERR(cudaFree(artist_row_d));
    H_ERR(cudaFree(artist_pref_d));
    H_ERR(cudaFree(art_conf_I_d));
    H_ERR(cudaFree(art_conf_d));

        int *mat1_7d, *mat2_7d, *res_7d;
            H_ERR(cudaMalloc((void **)&mat1_7d, sizeof(int) * endOfUserIndex * NUM_FEATURES));
            H_ERR(cudaMalloc((void **)&mat2_7d, sizeof(int) * NUM_FEATURES * NUM_FEATURES));
            H_ERR(cudaMalloc((void **)&res_7d, sizeof(int) * endOfUserIndex * NUM_FEATURES));

        int *mat1_8d, *mat2_8d, *res_8d;
            H_ERR(cudaMalloc((void **)&mat1_8d, sizeof(int) * endOfUserIndex * NUM_FEATURES));
            H_ERR(cudaMalloc((void **)&mat2_8d, sizeof(int) * endOfUserIndex * NUM_FEATURES));
            H_ERR(cudaMalloc((void **)&res_8d, sizeof(int) * endOfUserIndex * endOfUserIndex));

        int *mat1_add2, *mat2_add2, *res_add2;
        H_ERR(cudaMalloc((void **)&mat1_add2, sizeof(int) * endOfUserIndex * endOfUserIndex));
        H_ERR(cudaMalloc((void **)&mat2_add2, sizeof(int) * endOfUserIndex * endOfUserIndex));
        H_ERR(cudaMalloc((void **)&res_add2, sizeof(int) * endOfUserIndex * endOfUserIndex));

        int *mat1_9d, *mat2_9d, *res_9d;
            H_ERR(cudaMalloc((void **)&mat1_9d, sizeof(int) * endOfUserIndex * NUM_FEATURES));
            H_ERR(cudaMalloc((void **)&mat2_9d, sizeof(int) * NUM_FEATURES * NUM_FEATURES));
            H_ERR(cudaMalloc((void **)&res_9d, sizeof(int) * endOfUserIndex * NUM_FEATURES));

            int *mat1_10d, *mat2_10d, *res_10d;
            H_ERR(cudaMalloc((void **)&mat1_10d, sizeof(int) * endOfUserIndex * NUM_FEATURES));
            H_ERR(cudaMalloc((void **)&mat2_10d, sizeof(int) * endOfUserIndex * NUM_FEATURES));
            H_ERR(cudaMalloc((void **)&res_10d, sizeof(int) * endOfUserIndex * endOfUserIndex));

            H_ERR(cudaMemcpy(mat1_7d, X_T, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(mat2_8d, X, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(mat1_9d, X_T, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(mat2_10d, X, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));

            H_ERR(cudaMemcpy(mat1_add2, Y_P, sizeof(int) * endOfUserIndex * endOfUserIndex, cudaMemcpyHostToDevice));


        

        for(int j = 0; j < endOfArtistIndex; j++)
        {
            /*for(int k = 0; k < features; k++)
            {
                artist_row[k] = Y[j*features + k];
                if(artist_row[k] != 0)
                {
                    artist_pref[k] = 1;
                }
                else
                {
                    artist_pref[k] = artist_row[k];
                }
            }
            for(int k = 0; k < features; k++)
            {
                artist_confidence_I[k * features + k] = artist_row[k];
                artist_confidence[k * features + k] = artist_row[k] + 1;
            }*/
			//*********GPU*********//
            elapsed_time = wtime();
            elapsed_time -= time_beg;
            //printf("part 7 elapsed time is: %f\n", elapsed_time);

            time_beg = wtime();

            //mat_mat_multiply(X_T, artist_confidence_I, X_temp, endOfUserIndex, features, features);//

            
            H_ERR(cudaMemcpy(mat2_7d, artist_confidence_I + j * NUM_FEATURES, sizeof(int) * NUM_FEATURES * NUM_FEATURES, cudaMemcpyHostToDevice));

            gpu_mat_mat_multiply_atomic<<<256, 256>>>(mat1_7d, mat2_7d, res_7d, endOfUserIndex, NUM_FEATURES, NUM_FEATURES);

            H_ERR(cudaMemcpy(X_temp, res_7d, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyDeviceToHost));

            
            
            elapsed_time = wtime();
            elapsed_time -= time_beg;
            //printf("part 8 elapsed time is: %f\n", elapsed_time);

            time_beg = wtime();

            //mat_mat_multiply(X_temp, X, X_result_x, endOfUserIndex, features, endOfUserIndex);//

            
            H_ERR(cudaMemcpy(mat1_8d, X_temp, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
            
            gpu_mat_mat_multiply_atomic<<<256, 256>>>(mat1_8d, mat2_8d, res_8d, endOfUserIndex, NUM_FEATURES, endOfUserIndex);

            H_ERR(cudaMemcpy(X_result_x, res_8d, sizeof(int) * endOfUserIndex * endOfUserIndex, cudaMemcpyDeviceToHost));

            

			//*********GPU*********//
            /*for(int j = 0; j < endOfUserIndex; j++)
            {
                for(int k = 0; k < endOfUserIndex; k++)
                {
                    Y_result_y[j*endOfUserIndex + k] += Y_P[j*endOfUserIndex + k] + I1[j*endOfUserIndex + k];
                }
            }*/

            
            H_ERR(cudaMemcpy(mat2_add2, I1, sizeof(int) * endOfUserIndex * endOfUserIndex, cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(res_add2, Y_result_y, sizeof(int) * endOfUserIndex * endOfUserIndex, cudaMemcpyHostToDevice));

            gpu_matrix_addition<<<256, 256>>>(mat1_add2, mat2_add2, res_add2, endOfUserIndex, endOfUserIndex);

            H_ERR(cudaMemcpy(Y_result_y, res_add2, sizeof(int) * endOfUserIndex * endOfUserIndex, cudaMemcpyDeviceToHost));

            
			//*********GPU*********//

            //mat_mat_multiply(X_T, artist_confidence, X_temp_2, endOfUserIndex, features, features);//
            
            elapsed_time = wtime();
            elapsed_time -= time_beg;
            //printf("part 9 elapsed time is: %f\n", elapsed_time);

            time_beg = wtime();

            
            H_ERR(cudaMemcpy(mat2_9d, artist_confidence + j * NUM_FEATURES, sizeof(int) * NUM_FEATURES * NUM_FEATURES, cudaMemcpyHostToDevice));

            gpu_mat_mat_multiply_atomic<<<256, 256>>>(mat1_9d, mat2_9d, res_9d, endOfUserIndex, NUM_FEATURES, NUM_FEATURES);

            H_ERR(cudaMemcpy(X_temp_2, res_9d, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyDeviceToHost));       

             

            elapsed_time = wtime();
            elapsed_time -= time_beg;
            //printf("part 10 elapsed time is: %f\n", elapsed_time);

            time_beg = wtime();
            //mat_mat_multiply(X_temp_2, X, X_result_pi, endOfUserIndex, features, endOfUserIndex);//

            
            H_ERR(cudaMemcpy(mat1_10d, X_temp_2, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
            
            gpu_mat_mat_multiply_atomic<<<256, 256>>>(mat1_10d, mat2_10d, res_10d, endOfUserIndex, NUM_FEATURES, endOfUserIndex);

            H_ERR(cudaMemcpy(X_result_pi, res_10d, sizeof(int) * endOfUserIndex * endOfUserIndex, cudaMemcpyDeviceToHost));   

            

			//*********GPU*********//
            /*for(int k = 0; k < features; k++)
            {
                Y[i*features + k] = X_result_x[i*features + k] / X_result_pi[i*features + k];
            }*/

            
        }


        H_ERR(cudaFree(mat1_7d));
            H_ERR(cudaFree(mat2_7d));
            H_ERR(cudaFree(res_7d));

            H_ERR(cudaFree(mat1_8d));
            H_ERR(cudaFree(mat2_8d));
            H_ERR(cudaFree(res_8d));

            H_ERR(cudaFree(mat1_add2));
            H_ERR(cudaFree(mat2_add2));
            H_ERR(cudaFree(res_add2));

             H_ERR(cudaFree(mat1_9d));
            H_ERR(cudaFree(mat2_9d));
            H_ERR(cudaFree(res_9d));   

            H_ERR(cudaFree(mat1_10d));
            H_ERR(cudaFree(mat2_10d));
            H_ERR(cudaFree(res_10d));

        int *mat1_div, *mat2_div, *res_div;
        H_ERR(cudaMalloc((void **)&mat1_div, sizeof(int) * endOfUserIndex * NUM_FEATURES));
        H_ERR(cudaMalloc((void **)&mat2_div, sizeof(int) * endOfUserIndex * NUM_FEATURES));
        H_ERR(cudaMalloc((void **)&res_div, sizeof(int) * endOfUserIndex * endOfUserIndex));

        H_ERR(cudaMemcpy(mat1_div, Y_result_y, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
        H_ERR(cudaMemcpy(mat2_div, Y_result_pu, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice));

        gpu_mat_div<<<256, 256>>>(mat1_div, mat2_div, res_div, endOfUserIndex, NUM_FEATURES);

        H_ERR(cudaMemcpy(X, res_div, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyDeviceToHost));

        H_ERR(cudaFree(mat1_div));
        H_ERR(cudaFree(mat2_div));
        H_ERR(cudaFree(res_div));



        int *mat1_div2, *mat2_div2, *res_div2;
        H_ERR(cudaMalloc((void **)&mat1_div2, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
        H_ERR(cudaMalloc((void **)&mat2_div2, sizeof(int) * endOfArtistIndex * NUM_FEATURES));
        H_ERR(cudaMalloc((void **)&res_div2, sizeof(int) * endOfArtistIndex * endOfArtistIndex));

        H_ERR(cudaMemcpy(mat1_div2, X_result_x, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));
        H_ERR(cudaMemcpy(mat2_div2, X_result_pi, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice));

        gpu_mat_div<<<256, 256>>>(mat1_div2, mat2_div2, res_div2, endOfArtistIndex, NUM_FEATURES);

        H_ERR(cudaMemcpy(Y, res_div2, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyDeviceToHost));

        H_ERR(cudaFree(mat1_div2));
        H_ERR(cudaFree(mat2_div2));
        H_ERR(cudaFree(res_div2));
    }
    elapsed_time = wtime();
    elapsed_time -= time_beg;
    //printf("part 11 elapsed time is: %f\n", elapsed_time);


}

int main (int args, char **argv)
{
    
    double start_time = wtime();
    int newname = 0;
    dataMatrix = (int *)malloc(sizeof(int) * SPARSE_SIZE * SPARSE_SIZE);
    
	users = (char**)malloc(sizeof(char*) * USER_SIZE);
    for(int i = 0; i < USER_SIZE; i++)
    {
        users[i] = (char*)malloc(50 * sizeof(char));
    }
    artists = (char**)malloc(sizeof(char*) * ARTIST_SIZE);
    for(int i = 0; i < ARTIST_SIZE; i++)
    {
        artists[i] = (char*)malloc(50 * sizeof(char));
    }
    artistNames = (char**)malloc(sizeof(char*) * ARTIST_SIZE);
    for(int i = 0; i < ARTIST_SIZE; i++)
    {
        artistNames[i] = (char*)malloc(50 * sizeof(char));
    }
    FILE* data = fopen("usersha1-artmbid-artname-plays.tsv", "r"); //our dataset file (tab separated file)
	if(data == NULL)
	{
		//printf("File read error");
		return 0;
	}

	//j: 0 (user id), 1 (artist id), 2 (artist name), 3(plays)
	long i = 0;
    int j = 0;
    int currentUserIndex = 0, currentArtistIndex = 0, currentPlayCount = 0;
    while (1)
    {
        char dataLine[LINE_SIZE]; 
        if(i < INPUT_SIZE && fgets(dataLine, sizeof(dataLine), data) != NULL)//reading in entire line using fgets and putting it in dataLine
		{
			char * token = strtok(dataLine, "\t"); //parsing the data with the tab separater
			
			j = 0;
			while(j < 4) {
                if(token == NULL)
                {
                    break;
                }
                if(j == 0)//user id, check if its in the user list: if not, add to list, if it is, save the index
                {
                    currentUserIndex = checkIfUserExistsInData(token);
                    if(currentUserIndex == -1) //must add to users
                    {
                        currentUserIndex = endOfUserIndex;
                        strcpy(users[endOfUserIndex++], token);
                    }
                }
                else if (j == 1) //artist id, check if its in the artist list: if not, add to list, if it is, save the index
                {
                    newname = 0;
                    currentArtistIndex = checkIfArtistExistsInData(token);
                    if(currentArtistIndex == -1) //must add to artists
                    {
                        currentArtistIndex = endOfArtistIndex;
                        strcpy(artists[endOfArtistIndex], token);
                        newname = 1;
                    }
                }
                else if(j == 2)//artist name
                {
                    if(newname == 1)
                    strcpy(artistNames[endOfArtistIndex++], token);
                }
                else if(j == 3) //plays, use the indexes to see where they should go in the data (sparse matrix)
                {
                    currentPlayCount = atoi(token); //convert to integer and place in sparse matrix
                    dataMatrix[currentUserIndex * SPARSE_SIZE + currentArtistIndex] = currentPlayCount;
                }
				token = strtok(NULL, "\t"); //reading the next value of the parsed data
				j++;
			}
			i++;
		}
		else
		{
			break;
		}
    }

    int *ans;
    ans = (int *)malloc(sizeof(int) * NUM_RECOMMENDATIONS);
    double time_beg = wtime();
	implicit_als(40, ITERATIONS, 0.1, 10);
    double elapsed_time = wtime();
    elapsed_time -= time_beg;
    printf("implicit elapsed time is: %f\n", elapsed_time);
    time_beg = wtime();
    recommend(USER_ID, NUM_RECOMMENDATIONS, ans);
    elapsed_time = wtime();
    elapsed_time -= time_beg;
    printf("recommend elapsed time is: %f\n", elapsed_time);
    printf("User %d Recommendations: \n", USER_ID);
    for(int i = 0; i < NUM_RECOMMENDATIONS; i++)
    {
        printf("%s\n", artistNames[ans[i]]);
    }

    elapsed_time = wtime();
    elapsed_time -= start_time;
    printf("total elapsed time is: %f\n", elapsed_time);
    
	return 0;
}

