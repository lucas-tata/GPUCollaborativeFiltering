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

int main (int args, char **argv)
{

	FILE* data = fopen("usersha1-artmbid-artname-plays.tsv", "r"); //our dataset file (tab separated file)
	if(data == NULL)
	{
		printf("File read error");
		return 0;
	}

	char dataLine[1024]; //where we will store each line of the data read
	char * dataMatrix[1024][3]; //our output matrix we are storing in (basically a 2d string array)
	int i = 0, j = 0;
    while (1)
    {
        if(i < 1024 && fgets(dataLine, sizeof(dataLine), data) != NULL) //reading in 1024 lines using fgets and putting it in dataLine
		{
			char * token = strtok(dataLine, "\t"); //parsing the data with the tab separater
			
			j = 0;
			while(j < 3) {
				
				dataMatrix[i][j] = token;
				//printf( " %s\n", dataMatrix[i][j] );
				token = strtok(NULL, "\t"); //reading the next value of the parsed data
				if (token == NULL) //if there isnt a token, break out
				break;
				//printf("%s ", token);
				j++;
			}

			i++;
		}
		else
		{
			break;
		}
    }
	for(i = 0; i < 1024; i++)
	{
		for(j = 0; j < 3; j++)
		{
			printf("%s\n ", dataMatrix[i][j]); //as you can see from the print statements, data isn't being stored correctly
		}
	}
	
	return 0;
}

