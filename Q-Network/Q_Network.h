#include<iostream>
#include<stdio.h>
#include<random>
#include<ctime>
#include<cstdlib>
#include<algorithm>

using namespace std;

// Possible states in the space that can be visited by the query space
vector<vector<int>> possible_state_identifier()
{
    vector<vector<int>> ps{ {2, 4},
                            {1, 3, 5},
                            {2, 6},
                            {1, 5, 7},
                            {2, 4, 6, 8},
                            {3, 5, 9},
                            {4, 8},
                            {5, 7, 9},
                            {6, 8} }; 

    return ps;
}

// Possible actions in the space
vector<vector<int>> possible_action_identifier()
{
    vector<vector<int>> pa{ {1, 4}, // State 1 
                            {1, 3, 4}, // State 2
                            {3, 4}, // State 3
                            {1, 2, 4}, // State 4
                            {1, 2, 3, 4}, // State 5
                            {2, 3, 4}, // State 6
                            {1, 2}, // State 7
                            {1, 2, 3}, // State 8
                            {2, 3}}; // State 9

    return pa;
}

// Rewards for each s-a pair
vector<int> reward_identifier()
{
    // vector<int> rewards{10, -10, -10, -10, -100, -10, -25, -10, 10};
    vector<int> rewards{1, -1, -1, -1, -3, -1, 0, -1, 1};

    return rewards; 
}

// New state after taking action
int new_state_identifier(int st, int at)
{
    int s_tplus1;

    if (at == 1)
        s_tplus1 = st + 1;
    else if (at == 2)
        s_tplus1 = st - 3;
    else if (at == 3)
        s_tplus1 = st - 1;
    else if (at == 4)
        s_tplus1 = st + 3;

    return s_tplus1;
}

// Generates Episodes and stores experience replays
vector<vector<int>> experience_replay_generator(vector<vector<int>> possible_actions, vector<int> rewards, int number_of_episodes)
{   
    vector<vector<int>> experience_replays; // Stores the experience replays.
    for(int i=0; i<(2*number_of_episodes); i=i+2)
    {
        int st = 7, flag = 0;

        while(flag<2)
        {
            // Find st, at, r_t+1, s_t+1
            int rand_index = rand() % possible_actions[st-1].size();
            int at = possible_actions[st-1][rand_index];
            int s_tplus1 = new_state_identifier(st, at);
            int r_tplus1 = rewards[s_tplus1-1];

            // Add to experience replays
            vector<int> replay {st, at, r_tplus1, s_tplus1};
            experience_replays.push_back(replay);

            // Updating the state
            st = s_tplus1;

            flag = flag + 1;
        }
    }

    return experience_replays;
}

// The Q-Network inference
double QNetwork_inference(double* weight, int state)
{
    static double q_values[4];

    for(int i=0; i<4; i++)
    {
        double cal = weight[i]*state;
        q_values[i] = exp(cal)/(1 + exp(cal));
        cout<<q_values[i]<<", ";
    }
    cout<<"\n";

    double action_q_value = *max_element(q_values, q_values + sizeof(q_values) / sizeof(q_values[0]));

    // cout<<action_q_value<<"\n";

    return action_q_value;
}

// The Q-Network training
double* QNetwork_training(double* weight, double r_tplus1, double a_tplus1_qvalue, double at_qvalue)
{
    double q_star = r_tplus1 + a_tplus1_qvalue; // q*
    double q = at_qvalue; // q

    double lr = 0.001; // Learning rate
    double loss = q_star - q; // Loss

    int number_of_descent_steps = 25;
    for (int k=0; k<number_of_descent_steps; k++)
    {
        for(int i=0; i<4; i++)
        {
            weight[i] = weight[i] - lr*loss;
        }
    }

    return weight;
}