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

#define INPUT_SIZE 16384
#define SPARSE_SIZE 8192
#define USER_SIZE 2048
#define ARTIST_SIZE 8192
#define LINE_SIZE 1024


int *dataMatrix; //our output sparse matrix (users by artists, data is the play count) 
char **artists;
char **users;
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

int implicit_als(int alpha_val, int iterations, double lambda_val, int features)
{
    /*confidence = sparse_data * alpha_val
    
    # Get the size of user rows and item columns
    user_size, item_size = sparse_data.shape
    
    # We create the user vectors X of size users-by-features, the item vectors
    # Y of size items-by-features and randomly assign the values.
    X = sparse.csr_matrix(np.random.normal(size = (user_size, features)))
    Y = sparse.csr_matrix(np.random.normal(size = (item_size, features)))
    
    #Precompute I and lambda * I
    X_I = sparse.eye(user_size)
    Y_I = sparse.eye(item_size)
    
    I = sparse.eye(features)
    lI = lambda_val * I*/



    ///////////////////////////////////

    //calculate confidence
    //need to make another matrix?

    //////////////////////////////////

    //use endOfArtistIndex and endOfUserIndex to create X and Y

    /////////////////////////////////////

    //need to compute lambda and lambda * INPUT_SIZE

    ////////////////////////////////////

    
    
}

int main (int args, char **argv)
{
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
    printf("malloced!");
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
                    currentArtistIndex = checkIfArtistExistsInData(token);
                    if(currentArtistIndex == -1) //must add to artists
                    {
                        currentArtistIndex = endOfArtistIndex;
                        strcpy(artists[endOfArtistIndex++], token);
                    }
                }
                else if(j == 2)//artist name, doesnt really matter right now
                {
                    
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
	
    printf("artists size is %d ", endOfArtistIndex);
    printf("users size is %d ", endOfUserIndex);

    for(i = 0; i < endOfUserIndex; i++)
	{
        for(j = 0; j < endOfArtistIndex; j++)
        {
            printf("%d ", dataMatrix[i*SPARSE_SIZE + j]);
        }
        printf("\n");
	}
	implicit_als(40, 10, 0.1, 10);
	return 0;
}

