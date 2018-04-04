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

int *dataMatrix, *X, *Y, *X_T, *Y_T; //our output sparse matrix (users by artists, data is the play count) 
char **artists;
char **users;
char **artistNames;
int endOfArtistIndex = 0; //keep tabs on how many artists are currently in there
int endOfUserIndex = 0; //keep tabs on how many users are currently in there

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
            res += a[row * num_cols + j] * b[col + num_rows * j];
        }
        c[tid] = res;
        tid+= blockDim.x * gridDim.x;
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
	//*********GPU*********//

    for(int i = 0; i < num_items; i++)
    {
        maxVal = 0, index = 0;
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
    for(int i = 0; i < endOfArtistIndex; i++)
    {
        for(int j = 0; j < endOfUserIndex; j++)
        {
            dataMatrix[i * SPARSE_SIZE + j] = dataMatrix[i * SPARSE_SIZE + j] * alpha_val;
        }
    }
    
    int *X_P, *Y_P; 

    X = (int *)malloc(sizeof(int) * endOfUserIndex * NUM_FEATURES);
    Y = (int *)malloc(sizeof(int) * endOfArtistIndex * NUM_FEATURES);
    X_T = (int *)malloc(sizeof(int) * endOfUserIndex * NUM_FEATURES);
    Y_T = (int *)malloc(sizeof(int) * endOfArtistIndex * NUM_FEATURES);
    
    
    X_P = (int *)malloc(sizeof(int) * endOfUserIndex * endOfUserIndex);
    Y_P = (int *)malloc(sizeof(int) * endOfArtistIndex * endOfArtistIndex);
    
    for(int i = 0; i < endOfUserIndex; i++)
    {
        for(int j = 0; j < features; j++)
        {
            X[i * features + j] = rand() % RAND_RANGE;
        }
    }

    for(int i = 0; i < endOfUserIndex; i++)
    {
        for(int j = 0; j < features; j++)
        {
            X_T[j * endOfUserIndex + i] = X[i*features + j];
        }
    }




    for(int i = 0; i < endOfArtistIndex; i++)
    {
        for(int j = 0; j < features; j++)
        {
            Y[i * features + j] = rand() % RAND_RANGE;
        }
    }
    for(int i = 0; i < endOfArtistIndex; i++)
    {
        for(int j = 0; j < features; j++)
        {
            Y_T[j * endOfArtistIndex + i] = Y[i * features + j];
        }
    }


    int *X_I, *Y_I, *I, *I1, *user_row, *artist_row, *user_pref, *artist_pref, *user_confidence, *artist_confidence, *user_confidence_I, *artist_confidence_I; 

    X_I = (int *)malloc(sizeof(int) * endOfUserIndex * endOfUserIndex);
    Y_I = (int *)malloc(sizeof(int) * endOfArtistIndex * endOfArtistIndex);
    I = (int *)malloc(sizeof(int) * features * features);
    I1 = (int *)malloc(sizeof(int) * endOfArtistIndex * endOfArtistIndex);
    user_row = (int *)malloc(sizeof(int) * features);
    artist_row = (int *)malloc(sizeof(int) * features);
    user_pref = (int *)malloc(sizeof(int) * features);
    artist_pref = (int *)malloc(sizeof(int) * features);
    user_confidence = (int *)malloc(sizeof(int) * features * features);
    artist_confidence = (int *)malloc(sizeof(int) * features * features);
    user_confidence_I = (int *)malloc(sizeof(int) * features * features);
    artist_confidence_I = (int *)malloc(sizeof(int) * features * features);

    int *X_temp, *Y_temp, *Y_result_y, *Y_result_pu, *Y_temp_2, *X_result_x, *X_result_pi, *X_temp_2;
    X_temp = (int *)malloc(sizeof(int) * endOfUserIndex * features);
    X_temp_2 = (int *)malloc(sizeof(int) * endOfUserIndex * features);
    X_result_x = (int *)malloc(sizeof(int) * endOfUserIndex * endOfUserIndex);
    X_result_pi = (int *)malloc(sizeof(int) * endOfUserIndex * endOfUserIndex);
    Y_temp = (int *)malloc(sizeof(int) * endOfArtistIndex * features);
    Y_temp_2 = (int *)malloc(sizeof(int) * endOfArtistIndex * features);
    Y_result_y = (int *)malloc(sizeof(int) * endOfArtistIndex * endOfArtistIndex);
    Y_result_pu = (int *)malloc(sizeof(int) * endOfArtistIndex * endOfArtistIndex);

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

    //mat_mat_multiply(X, X_T, X_P, endOfUserIndex, features, endOfUserIndex);//
    int *mat1_d, *mat2_d, *res_d;
    cudaMalloc((void **)&mat1_d, sizeof(int) * endOfUserIndex * NUM_FEATURES);
    cudaMalloc((void **)&mat2_d, sizeof(int) * endOfUserIndex * NUM_FEATURES);
    cudaMalloc((void **)&res_d, sizeof(int) * endOfUserIndex * endOfUserIndex);

    cudaMemcpy(mat1_d, X, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
    cudaMemcpy(mat2_d, X_T, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice);

    gpu_mat_mat_multiply<<<256, 256>>>(mat1_d, mat2_d, res_d, endOfUserIndex, NUM_FEATURES, endOfUserIndex);

    cudaMemcpy(X_P, res_d, sizeof(int) * endOfUserIndex * endOfUserIndex, cudaMemcpyDeviceToHost);

    
    //mat_mat_multiply(Y, Y_T, Y_P, endOfArtistIndex, features, endOfArtistIndex);//
    int *mat1_2d, *mat2_2d, *res_2d;
    cudaMalloc((void **)&mat1_2d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);
    cudaMalloc((void **)&mat2_2d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);
    cudaMalloc((void **)&res_2d, sizeof(int) * endOfArtistIndex * endOfArtistIndex);

    cudaMemcpy(mat1_2d, Y, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
    cudaMemcpy(mat2_2d, Y_T, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice);

    gpu_mat_mat_multiply<<<256, 256>>>(mat1_2d, mat2_2d, res_2d, endOfArtistIndex, NUM_FEATURES, endOfArtistIndex);

    cudaMemcpy(Y_P, res_2d, sizeof(int) * endOfArtistIndex * endOfArtistIndex, cudaMemcpyDeviceToHost);

	//*********GPU*********//


    for(int i = 0; i < iterations; i++)
    {
        for(int j = 0; j < endOfUserIndex; j++)
        {
            for(int k = 0; k < features; k++)
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
            }

            //*********GPU*********//
            // mat_mat_multiply(Y_T, user_confidence_I, Y_temp, endOfArtistIndex, features, features);//
            int *mat1_3d, *mat2_3d, *res_3d;
            cudaMalloc((void **)&mat1_3d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);
            cudaMalloc((void **)&mat2_3d, sizeof(int) * NUM_FEATURES * NUM_FEATURES);
            cudaMalloc((void **)&res_3d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);

            cudaMemcpy(mat1_3d, Y_T, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
            cudaMemcpy(mat2_3d, user_confidence_I, sizeof(int) * NUM_FEATURES * NUM_FEATURES, cudaMemcpyHostToDevice);

            gpu_mat_mat_multiply<<<256, 256>>>(mat1_3d, mat2_3d, res_3d, endOfArtistIndex, NUM_FEATURES, NUM_FEATURES);

            cudaMemcpy(Y_temp, res_3d, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyDeviceToHost);
           
            //mat_mat_multiply(Y_temp, Y, Y_result_y, endOfArtistIndex, features, endOfArtistIndex);//

            int *mat1_4d, *mat2_4d, *res_4d;
            cudaMalloc((void **)&mat1_4d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);
            cudaMalloc((void **)&mat2_4d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);
            cudaMalloc((void **)&res_4d, sizeof(int) * endOfArtistIndex * endOfArtistIndex);

            cudaMemcpy(mat1_4d, Y_temp, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
            cudaMemcpy(mat2_4d, Y, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice);

            gpu_mat_mat_multiply<<<256, 256>>>(mat1_4d, mat2_4d, res_4d, endOfArtistIndex, NUM_FEATURES, endOfArtistIndex);

            cudaMemcpy(Y_result_y, res_4d, sizeof(int) * endOfArtistIndex * endOfArtistIndex, cudaMemcpyDeviceToHost);
			//*********GPU*********//
            for(int j = 0; j < endOfArtistIndex; j++)
            {
                for(int k = 0; k < endOfArtistIndex; k++)
                {
                    Y_result_y[j*endOfArtistIndex + k] += Y_P[j*endOfArtistIndex + k] + I1[j*endOfArtistIndex + k];
                }
            }
			//*********GPU*********//
            //mat_mat_multiply(Y_T, user_confidence, Y_temp_2, endOfArtistIndex, features, features);//
            int *mat1_5d, *mat2_5d, *res_5d;
            cudaMalloc((void **)&mat1_5d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);
            cudaMalloc((void **)&mat2_5d, sizeof(int) * NUM_FEATURES * NUM_FEATURES);
            cudaMalloc((void **)&res_5d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);

            cudaMemcpy(mat1_5d, Y_T, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
            cudaMemcpy(mat2_5d, user_confidence, sizeof(int) * NUM_FEATURES * NUM_FEATURES, cudaMemcpyHostToDevice);

            gpu_mat_mat_multiply<<<256, 256>>>(mat1_5d, mat2_5d, res_5d, endOfArtistIndex, NUM_FEATURES, NUM_FEATURES);

            cudaMemcpy(Y_temp_2, res_5d, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyDeviceToHost);

            //mat_mat_multiply(Y_temp_2, Y, Y_result_pu, endOfArtistIndex, features, endOfArtistIndex);//

            int *mat1_6d, *mat2_6d, *res_6d;
            cudaMalloc((void **)&mat1_6d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);
            cudaMalloc((void **)&mat2_6d, sizeof(int) * endOfArtistIndex * NUM_FEATURES);
            cudaMalloc((void **)&res_6d, sizeof(int) * endOfArtistIndex * endOfArtistIndex);

            cudaMemcpy(mat1_6d, Y_temp_2, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
            cudaMemcpy(mat2_6d, Y, sizeof(int) * endOfArtistIndex * NUM_FEATURES, cudaMemcpyHostToDevice);

            gpu_mat_mat_multiply<<<256, 256>>>(mat1_6d, mat2_6d, res_6d, endOfArtistIndex, NUM_FEATURES, endOfArtistIndex);

            cudaMemcpy(Y_result_pu, res_6d, sizeof(int) * endOfArtistIndex * endOfArtistIndex, cudaMemcpyDeviceToHost);

			//*********GPU*********//
            for(int k = 0; k < features; k++)
            {
                X[i*features + k] = Y_result_y[i*features + k] / Y_result_pu[i*features + k];
            }
        }
        for(int j = 0; j < endOfArtistIndex; j++)
        {
            for(int k = 0; k < features; k++)
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
            }
			//*********GPU*********//

            //mat_mat_multiply(X_T, artist_confidence_I, X_temp, endOfUserIndex, features, features);//

            int *mat1_7d, *mat2_7d, *res_7d;
            cudaMalloc((void **)&mat1_7d, sizeof(int) * endOfUserIndex * NUM_FEATURES);
            cudaMalloc((void **)&mat2_7d, sizeof(int) * NUM_FEATURES * NUM_FEATURES);
            cudaMalloc((void **)&res_7d, sizeof(int) * endOfUserIndex * NUM_FEATURES);

            cudaMemcpy(mat1_7d, X_T, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
            cudaMemcpy(mat2_7d, artist_confidence_I, sizeof(int) * NUM_FEATURES * NUM_FEATURES, cudaMemcpyHostToDevice);

            gpu_mat_mat_multiply<<<256, 256>>>(mat1_7d, mat2_7d, res_7d, endOfUserIndex, NUM_FEATURES, NUM_FEATURES);

            cudaMemcpy(X_temp, res_7d, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyDeviceToHost);


            //mat_mat_multiply(X_temp, X, X_result_x, endOfUserIndex, features, endOfUserIndex);//

            int *mat1_8d, *mat2_8d, *res_8d;
            cudaMalloc((void **)&mat1_8d, sizeof(int) * endOfUserIndex * NUM_FEATURES);
            cudaMalloc((void **)&mat2_8d, sizeof(int) * endOfUserIndex * NUM_FEATURES);
            cudaMalloc((void **)&res_8d, sizeof(int) * endOfUserIndex * endOfUserIndex);

            cudaMemcpy(mat1_8d, X_temp, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
            cudaMemcpy(mat2_8d, X, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice);

            gpu_mat_mat_multiply<<<256, 256>>>(mat1_8d, mat2_8d, res_8d, endOfUserIndex, NUM_FEATURES, endOfUserIndex);

            cudaMemcpy(X_result_x, res_8d, sizeof(int) * endOfUserIndex * endOfUserIndex, cudaMemcpyDeviceToHost);
			//*********GPU*********//
            for(int j = 0; j < endOfUserIndex; j++)
            {
                for(int k = 0; k < endOfUserIndex; k++)
                {
                    Y_result_y[j*endOfUserIndex + k] += Y_P[j*endOfUserIndex + k] + I1[j*endOfUserIndex + k];
                }
            }
			//*********GPU*********//

            //mat_mat_multiply(X_T, artist_confidence, X_temp_2, endOfUserIndex, features, features);//

            int *mat1_9d, *mat2_9d, *res_9d;
            cudaMalloc((void **)&mat1_9d, sizeof(int) * endOfUserIndex * NUM_FEATURES);
            cudaMalloc((void **)&mat2_9d, sizeof(int) * NUM_FEATURES * NUM_FEATURES);
            cudaMalloc((void **)&res_9d, sizeof(int) * endOfUserIndex * NUM_FEATURES);

            cudaMemcpy(mat1_9d, X_T, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
            cudaMemcpy(mat2_9d, artist_confidence, sizeof(int) * NUM_FEATURES * NUM_FEATURES, cudaMemcpyHostToDevice);

            gpu_mat_mat_multiply<<<256, 256>>>(mat1_9d, mat2_9d, res_9d, endOfUserIndex, NUM_FEATURES, NUM_FEATURES);

            cudaMemcpy(X_temp_2, res_9d, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyDeviceToHost);            


            //mat_mat_multiply(X_temp_2, X, X_result_pi, endOfUserIndex, features, endOfUserIndex);//

            int *mat1_10d, *mat2_10d, *res_10d;
            cudaMalloc((void **)&mat1_10d, sizeof(int) * endOfUserIndex * NUM_FEATURES);
            cudaMalloc((void **)&mat2_10d, sizeof(int) * endOfUserIndex * NUM_FEATURES);
            cudaMalloc((void **)&res_10d, sizeof(int) * endOfUserIndex * endOfUserIndex);

            cudaMemcpy(mat1_10d, X_temp_2, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice);
            cudaMemcpy(mat2_10d, X, sizeof(int) * endOfUserIndex * NUM_FEATURES, cudaMemcpyHostToDevice);

            gpu_mat_mat_multiply<<<256, 256>>>(mat1_10d, mat2_10d, res_10d, endOfUserIndex, NUM_FEATURES, endOfUserIndex);

            cudaMemcpy(X_result_pi, res_10d, sizeof(int) * endOfUserIndex * endOfUserIndex, cudaMemcpyDeviceToHost);   

			//*********GPU*********//
            for(int k = 0; k < features; k++)
            {
                if(X_result_pi[i*features + k] == 0)
                    Y[i*features + k] = 0;
                else
                    Y[i*features + k] = X_result_x[i*features + k] / X_result_pi[i*features + k];
            }
        }
        
    }


}

int main (int args, char **argv)
{
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
		printf("File read error");
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
	implicit_als(40, ITERATIONS, 0.1, 10);
    recommend(USER_ID, NUM_RECOMMENDATIONS, ans);
    printf("User %d Recommendations: \n", USER_ID);
    for(int i = 0; i < NUM_RECOMMENDATIONS; i++)
    {
        printf("%s\n", artistNames[ans[i]]);
    }
	return 0;
}

