#include <iostream>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <vector>
#include <random>
#include <string>

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#define MAXHOSTNAME 256
gsl_rng * r;

using namespace std;

struct distribution
{
  int alpha;
  int beta;
  distribution()
  {
    alpha = 1;beta=1;
  }
};

void options(){

  cout << "Usage:\n";
  cout << "bandit-agent\n"; 
  cout << "\t[--numArms numArms]\n";
  cout << "\t[--randomSeed randomSeed]\n";
  cout << "\t[--horizon horizon]\n";
  cout << "\t[--hostname hostname]\n";
  cout << "\t[--port port]\n";
  cout << "\t[--algorithm algorithm]\n";
  cout << "\t[--epsilon epsilon]\n";

}


/*
  Read command line arguments, and set the ones that are passed (the others remain default.)
*/
bool setRunParameters(int argc, char *argv[], int &numArms, int &randomSeed, unsigned long int &horizon, string &hostname, int &port, string &algorithm, double &epsilon){

  int ctr = 1;
  while(ctr < argc){

    //cout << string(argv[ctr]) << "\n";

    if(string(argv[ctr]) == "--help"){
      return false;//This should print options and exit.
    }
    else if(string(argv[ctr]) == "--numArms"){
      if(ctr == (argc - 1)){
	return false;
      }
      numArms = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--randomSeed"){
      if(ctr == (argc - 1)){
	return false;
      }
      randomSeed = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--horizon"){
      if(ctr == (argc - 1)){
	return false;
      }
      horizon = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--hostname"){
      if(ctr == (argc - 1)){
	return false;
      }
      hostname = string(argv[ctr + 1]);
      ctr++;
    }
    else if(string(argv[ctr]) == "--port"){
      if(ctr == (argc - 1)){
	return false;
      }
      port = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--algorithm"){
      if(ctr == (argc - 1)){
  return false;
      }
      algorithm = string(argv[ctr + 1]);
      ctr++;
    }
     else if(string(argv[ctr]) == "--epsilon"){
      if(ctr == (argc - 1)){
  return false;
      }
      epsilon = atof(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else{
      return false;
    }

    ctr++;
  }

  return true;
}

/* The KL Divergence function */
double KL(double p, double q)
{
  double sum=0.0;
  sum = sum + p*log(p/q);
  p = 1-p;q = 1-q;
  sum = sum + p*log(p/q);
  return sum;
}

/* Newton iterations method to determine q for KL-UCB */
double newton_find(double low,double high,double bound)
{
  double mid,value,epsilon,avg;
  avg = low;
  epsilon = 0.001;
  do
  {
      mid = (low+high)/2;
      value = KL(avg,mid);
      if(value>bound)
      {
          high = mid;
      }
      else
      {
          low = mid;
      }
  }
  while(abs(value-bound)>epsilon);

  return mid;
}

/* ============================================================================= */
/* Write your algorithms here */
int sampleArm(string algorithm, double epsilon, int pulls, float reward, int numArms, float * my_rewards, int * my_pulls, distribution * my_distribution){
  if(algorithm.compare("rr") == 0){
    return(pulls % numArms);
  }
  else if(algorithm.compare("epsilon-greedy") == 0)
        {
          /* Write e-greedy algorithm here */
          bool random_move = (rand() % 100) < (epsilon * 100);
          if(random_move)
          {
              int gen = rand() % numArms;
              return gen;
          }
          else
          {
              double max = -1.0;double temp;
              int max_index = -1;
              for(int i=0; i<numArms; i++)
              {
                  if(!(my_pulls[i]==0)) temp = my_rewards[i]/my_pulls[i];
                  else temp = 0.0;
                  if(temp>max){max = temp;max_index = i;}
                  else
                  {
                    if(temp==max){if(my_pulls[i]==0) max_index = i;}
                  }
              }

              return max_index;
          }

          /* e-greedy end */
        }
  else if(algorithm.compare("UCB") == 0)
        {
          /* Write UCB algorithm here */
          /* check if all arms are pulled atleast once */
          bool all_pulled = 1;int i=0;
          for( ;i<numArms;i++)
          {
            if(my_pulls[i]==0){all_pulled=0;break;}
          }

          if(!all_pulled)
          {
                /* If all arms are not pulled, pull any non pulled arm */
                return i;
          }
          else
          {
                double max = -1.0;double temp,average,offset;
                int max_index = -1;
                for(int i=0; i<numArms; i++)
                {
                    average = my_rewards[i]/my_pulls[i];
                    offset = sqrt( (2*log(pulls))/my_pulls[i] );
                    temp = average + offset;
                    if(temp>max){max = temp;max_index = i;}
                }

                return max_index;
          }
          /* UCB end */
        }
  else if(algorithm.compare("KL-UCB") == 0)
        {
           /* Write KL-UCB algorithm here */
           /* check if all arms are pulled once */
              bool all_pulled = 1;int i=0;
              for( ;i<numArms;i++)
              {
                if(my_pulls[i]==0){all_pulled=0;break;}
              }

              if(!all_pulled)
              {
                    /* If all arms are not pulled, pull any non pulled arm */
                    return i;
              }
              else
              {
                    double max = -1.0;double average,maximum,bound,ucb_kl;
                    int constant = 0;
                    maximum = 1.0;
                    int max_index = -1;
                    for(int i=0; i<numArms; i++)
                    {
                        average = my_rewards[i]/my_pulls[i];
                        bound = (log(pulls)+constant * log(log(pulls)))/my_pulls[i];
                        ucb_kl = newton_find(average,maximum,bound);
                        if(ucb_kl>max){max = ucb_kl;max_index = i;}
                    }

                    return max_index;
              }
           /* KL-UCB end */
        }
  else if(algorithm.compare("Thompson-Sampling") == 0)
        {
          /* Write Thompson Sampling algo here */
          // gsl_sf_beta(a,b)
          double max,temp;int max_index = -1;
          max = -1.0;
          for(int i=0;i<numArms;i++)
          {
            // temp = gsl_sf_beta(my_distribution[i].alpha, my_distribution[i].beta);
            temp = gsl_ran_beta (r, my_distribution[i].alpha, my_distribution[i].beta);
            if(temp>max){max = temp;max_index = i;}
          }

          return max_index;

          /* Thompson sampling ends here */
        }
  else{
    return -1;
  }
  
}

/* ============================================================================= */


int main(int argc, char *argv[]){
  // Run Parameter defaults.
  int numArms = 5;
  int randomSeed = time(0);
  unsigned long int horizon = 200;
  string hostname = "localhost";
  int port = 5000;
  string algorithm="random";
  double epsilon=0.0;

  //Set from command line, if any.
  if(!(setRunParameters(argc, argv, numArms, randomSeed, horizon, hostname, port, algorithm, epsilon))){
    //Error parsing command line.
    options();
    return 1;
  }

  struct sockaddr_in remoteSocketInfo;
  struct hostent *hPtr;
  int socketHandle;

  bzero(&remoteSocketInfo, sizeof(sockaddr_in));
  
  if((hPtr = gethostbyname((char*)(hostname.c_str()))) == NULL){
    cerr << "System DNS name resolution not configured properly." << "\n";
    cerr << "Error number: " << ECONNREFUSED << "\n";
    exit(EXIT_FAILURE);
  }

  if((socketHandle = socket(AF_INET, SOCK_STREAM, 0)) < 0){
    close(socketHandle);
    exit(EXIT_FAILURE);
  }

  memcpy((char *)&remoteSocketInfo.sin_addr, hPtr->h_addr, hPtr->h_length);
  remoteSocketInfo.sin_family = AF_INET;
  remoteSocketInfo.sin_port = htons((u_short)port);

  if(connect(socketHandle, (struct sockaddr *)&remoteSocketInfo, sizeof(sockaddr_in)) < 0){
    //code added
    cout<<"connection problem"<<".\n";
    close(socketHandle);
    exit(EXIT_FAILURE);
  }


  char sendBuf[256];
  char recvBuf[256];

  float reward = 0;
  unsigned long int pulls=0;
  int * my_pulls = new int[numArms];
  float * my_rewards  = new float[numArms];
  distribution * my_distribution = new distribution[numArms];

  const gsl_rng_type * T;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set(r, randomSeed);

  for(int i=0; i<numArms ; i++)
  {
    my_pulls[i] = 0;
    my_rewards[i] = 0.0;
  }

  int armToPull = sampleArm(algorithm, epsilon, pulls, reward, numArms, my_rewards, my_pulls, my_distribution);
  
  sprintf(sendBuf, "%d", armToPull);

  cout << "Sending action " << armToPull << ".\n";
  while(send(socketHandle, sendBuf, strlen(sendBuf)+1, MSG_NOSIGNAL) >= 0){

    char temp;
    recv(socketHandle, recvBuf, 256, 0);
    sscanf(recvBuf, "%f %c %lu", &reward, &temp, &pulls);
    cout << "Received reward " << reward << ".\n";
    cout<<"Num of  pulls "<<pulls<<".\n";

    my_pulls[armToPull] = my_pulls[armToPull] + 1;
    my_rewards[armToPull] = my_rewards[armToPull] + reward;
    my_distribution[armToPull].alpha = my_distribution[armToPull].alpha + reward;
    my_distribution[armToPull].beta = my_distribution[armToPull].beta + 1 - reward;

    armToPull = sampleArm(algorithm, epsilon, pulls, reward, numArms, my_rewards, my_pulls, my_distribution);

    sprintf(sendBuf, "%d", armToPull);
    cout << "Sending action " << armToPull << ".\n";
  }
  
  close(socketHandle);
  gsl_rng_free (r);
  cout << "Terminating.\n";

  return 0;
}
          
