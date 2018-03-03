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

#define INPUT_SIZE 64
#define LINE_SIZE 1024

int dataMatrix[INPUT_SIZE][INPUT_SIZE]; //our output sparse matrix (users by artists, data is the play count) 
char artists[INPUT_SIZE][50]; //running list of the different artists
char users[INPUT_SIZE][50]; //running list of the different users
int endOfArtistIndex = 0; //keep tabs on how many artists are currently in there
int endOfUserIndex = 0; //keep tabs on how many users are currently in there

int checkIfArtistExistsInData(char * artist)
{
    int i;
    for(i = 0; i < INPUT_SIZE; i++)
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
    for(i = 0; i < INPUT_SIZE; i++)
    {
        if(strcmp(user, users[i]) == 0)
        {
            return i;
        }
    }
    return -1;
}

int main (int args, char **argv)
{

	FILE* data = fopen("usersha1-artmbid-artname-plays.tsv", "r"); //our dataset file (tab separated file)
	if(data == NULL)
	{
		printf("File read error");
		return 0;
	}

	//j: 0 (user id), 1 (artist id), 2 (artist name), 3(plays)
	int i = 0, j = 0;
    int currentUserIndex = 0, currentArtistIndex = 0, currentPlayCount = 0;
    while (1)
    {
        char dataLine[LINE_SIZE]; 
        if(i < INPUT_SIZE && fgets(dataLine, sizeof(dataLine), data) != NULL) //reading in 1024 lines using fgets and putting it in dataLine
		{
			char * token = strtok(dataLine, "\t"); //parsing the data with the tab separater
			
			j = 0;
			while(j < 4) {
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
                    dataMatrix[currentUserIndex][currentArtistIndex] = currentPlayCount;
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
	for(i = 0; i < INPUT_SIZE; i++)
	{
        for(j = 0; j < INPUT_SIZE; j++)
        {
            printf("%d ", dataMatrix[i][j]);
        }
        printf("\n");
	}
	
	return 0;
}

