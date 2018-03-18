/* 
Lucas Tata / Salvatore Amico
High Performance Computing on GPUs
Final Project
Collaborative Filtering
CPU Implementation
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

    mat_vec_multiply(Y_T, X_rec, rec_vector, NUM_FEATURES, endOfArtistIndex);

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

    mat_mat_multiply(X, X_T, X_P, endOfUserIndex, features, endOfUserIndex);
    mat_mat_multiply(Y, Y_T, Y_P, endOfArtistIndex, features, endOfArtistIndex);


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

            
            mat_mat_multiply(Y_T, user_confidence_I, Y_temp, endOfArtistIndex, features, features);
            mat_mat_multiply(Y_temp, Y, Y_result_y, endOfArtistIndex, features, endOfArtistIndex);
            for(int j = 0; j < endOfArtistIndex; j++)
            {
                for(int k = 0; k < endOfArtistIndex; k++)
                {
                    Y_result_y[j*endOfArtistIndex + k] += Y_P[j*endOfArtistIndex + k] + I1[j*endOfArtistIndex + k];
                }
            }

            mat_mat_multiply(Y_T, user_confidence, Y_temp_2, endOfArtistIndex, features, features);
            mat_mat_multiply(Y_temp_2, Y, Y_result_pu, endOfArtistIndex, features, endOfArtistIndex);
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
            mat_mat_multiply(X_T, artist_confidence_I, X_temp, endOfUserIndex, features, features);
            mat_mat_multiply(X_temp, X, X_result_x, endOfUserIndex, features, endOfUserIndex);
            for(int j = 0; j < endOfUserIndex; j++)
            {
                for(int k = 0; k < endOfUserIndex; k++)
                {
                    Y_result_y[j*endOfUserIndex + k] += Y_P[j*endOfUserIndex + k] + I1[j*endOfUserIndex + k];
                }
            }

            mat_mat_multiply(X_T, artist_confidence, X_temp_2, endOfUserIndex, features, features);
            mat_mat_multiply(X_temp_2, X, X_result_pi, endOfUserIndex, features, endOfUserIndex);
            for(int k = 0; k < features; k++)
            {
                Y[i*features + k] = X_result_x[i*features + k] / X_result_pi[i*features + k];
            }
        }
        
    }


}

int main (int args, char **argv)
{
    int newname = 0;
    dataMatrix = (int *)malloc(sizeof(int) * SPARSE_SIZE * SPARSE_SIZE);
    
	users = malloc(sizeof(char*) * USER_SIZE);
    for(int i = 0; i < USER_SIZE; i++)
    {
        users[i] = malloc(50 * sizeof(char));
    }
    artists = malloc(sizeof(char*) * ARTIST_SIZE);
    for(int i = 0; i < ARTIST_SIZE; i++)
    {
        artists[i] = malloc(50 * sizeof(char));
    }
    artistNames = malloc(sizeof(char*) * ARTIST_SIZE);
    for(int i = 0; i < ARTIST_SIZE; i++)
    {
        artistNames[i] = malloc(50 * sizeof(char));
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

