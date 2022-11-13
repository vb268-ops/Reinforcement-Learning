// A Q-Network being trained to obtain policy for a bear to reach the fish in the map.
// check "Map.png" for the map (in GitHub repository).

#include<iostream>
#include<stdio.h>
#include<random>
#include<ctime>
#include<cstdlib>

#include"Q_Network.h"

using namespace std;

int main()
{
    double QNetwork_weights[4] = {0.01, 0.01, 0.01, 0.01};

    // Identifying possible actions in each state using the attached .txt file.
    vector<vector<int>> possible_actions, possible_states;
    possible_actions = possible_action_identifier();
    possible_states = possible_state_identifier();

    vector<int> rewards;
    rewards = reward_identifier();

    // Generating Experience Replays for Training Q Network
    int number_of_episodes = 50;
    vector<vector<int>> experience_replays;
    experience_replays = experience_replay_generator(possible_actions, rewards, number_of_episodes);

    // Training Q Network
    double TargetNetwork_weights[4];
    memcpy(TargetNetwork_weights, QNetwork_weights, sizeof(TargetNetwork_weights));

    for(int i = 0; i < number_of_episodes; i++)
    {
        // Inference
        double at_qvalue = QNetwork_inference(QNetwork_weights, experience_replays[i][0]);
        double a_tplus1_qvalue = QNetwork_inference(TargetNetwork_weights, experience_replays[i][3]);

        // Training
        double* updated_weights;
        updated_weights = QNetwork_training(QNetwork_weights, experience_replays[i][2], a_tplus1_qvalue, at_qvalue);

        memcpy(QNetwork_weights, updated_weights, sizeof(updated_weights));

        // Updating Target Network weight at certain intervals
        if (i%5 == 0)
            memcpy(TargetNetwork_weights, QNetwork_weights, sizeof(TargetNetwork_weights));
    }

    double query_q = QNetwork_inference(QNetwork_weights, 4);
    cout<<"Final "<<query_q;
    return 0;
}
