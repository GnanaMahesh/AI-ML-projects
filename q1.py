import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt


''' Do not change anything in this function '''
def generate_random_profiles(num_voters, num_candidates):
    '''
        Generates a NumPy array where row i denotes the strict preference order of voter i
        The first value in row i denotes the candidate with the highest preference
        Result is a NumPy array of size (num_voters x num_candidates)
    '''
    return np.array([np.random.permutation(np.arange(1, num_candidates+1)) 
            for _ in range(num_voters)])


def find_winner(profiles, voting_rule):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        In STV, if there is a tie amongst the candidates with minimum plurality score in a round, then eliminate the candidate with the lower index
        For Copeland rule, ties among pairwise competitions lead to half a point for both candidates in their Copeland score

        Return: Index of winning candidate (1-indexed) found using the given voting rule
        If there is a tie amongst the winners, then return the winner with a lower index
    '''

    winner_index = None
    
    # TODO

    if voting_rule == 'plurality':
        toppref= np.array(profiles[:,0])
        contestants, votes = np.unique(toppref, return_counts=True)
        max_index=0
        for i in range(len(votes)):
          if votes[i] > votes[max_index]:
              max_index=i
          elif votes[i] == votes[max_index]:
              if(contestants[max_index] > contestants[i]):
                  max_index=i
        winner_index = contestants[max_index]

    elif voting_rule == 'borda':
        nvoters,ncontestants = profiles.shape
        bordascore=np.zeros(ncontestants)
        for i in range(ncontestants):
            for j in range(nvoters):
               bordascore[profiles[j,i]-1] = bordascore[profiles[j,i]-1]+ncontestants-i-1
        winner_index=np.argmax(bordascore)
        winner_index = winner_index+1

    elif voting_rule == 'stv':
        nvoters,ncontestants = profiles.shape
        eliminated = np.zeros(ncontestants, dtype=int)
        for i in range(ncontestants-1):
            toppref=np.zeros(nvoters, dtype=int)
            for vo in range(nvoters):
                for co in range(ncontestants):
                    if(eliminated[profiles[vo,co]-1] == 0):
                        toppref[vo] = profiles[vo,co]
                        break
            votes = np.zeros(ncontestants, dtype=int)
            for l in range(nvoters):
                votes[toppref[l]-1] = votes[toppref[l]-1]+1
            for j in range(ncontestants):
              if eliminated[j]==0:
                eliminate_index=j
                break
            for j in range(ncontestants):
              if votes[j] < votes[eliminate_index] and eliminated[j]==0:
                eliminate_index=j
            eliminated[eliminate_index]=1
        
        for i in range(ncontestants):
            if eliminated[i] == 0:
                winner_index = i+1

    elif voting_rule == 'copeland':
        nvoters,ncontestants = profiles.shape
        copeland = np.zeros(ncontestants, dtype=int)
        for c in range(ncontestants):
            better = np.zeros(ncontestants, dtype=int)
            for i in range(nvoters):
                for j in range(ncontestants):
                    if profiles[i,j] != c+1 :
                        better[profiles[i,j]-1] = better[profiles[i,j]-1]+1
                    else:
                        break
            for i in range(ncontestants):
                if ((2 * better[i]) < nvoters):
                    copeland[c]=copeland[c]+2
                elif ((2 * better[i]) == nvoters):
                    copeland[c]=copeland[c]+1
        winner_index = np.argmax(copeland)           
        winner_index = winner_index+1

    # END TODO

    return winner_index


def find_winner_average_rank(profiles, winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        winner is the index of the winning candidate for some voting rule (1-indexed)

        Return: The average rank of the winning candidate (rank wrt a voter can be from 1 to num_candidates)
    '''

    average_rank = None

    # TODO
    average_rank =0
    nvoters,ncontestants = profiles.shape
    for i in range(nvoters):
        for j in range(ncontestants):
            if profiles[i,j] == winner :
                average_rank = average_rank+j+1
    average_rank = average_rank/nvoters
    # END TODO

    return average_rank


def check_manipulable(profiles, voting_rule, find_winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        find_winner is a function that takes profiles and voting_rule as input, and gives the winner index as the output
        It is guaranteed that there will be at most 8 candidates if checking manipulability of a voting rule

        Return: Boolean representing whether the voting rule is manipulable for the given preference profiles
    '''

    manipulable = None

    # TODO
    nvoters,ncontestants = profiles.shape
    winner = find_winner(profiles,voting_rule)
    manipulable = False
    for i in range(nvoters):
        if(manipulable == True):
            break
        profiles1=profiles.copy()
        perms = list(itertools.permutations(np.array(profiles[i,:])))
        permut = np.array(perms)
        rows,colums = permut.shape
        for j in range(rows):
            profiles1[i,:]=np.array(permut[j,:])
            winner1 = find_winner(profiles1,voting_rule)
            index1 = np.where(np.array(profiles[i,:]) == winner1)[0]
            index = np.where(np.array(profiles[i,:]) == winner)[0]
            if(index1 < index):
                manipulable=True
                break
    # END TODO

    return manipulable


if __name__ == '__main__':
    np.random.seed(420)

    num_tests = 200
    voting_rules = ['plurality', 'borda', 'stv', 'copeland']

    average_ranks = [[] for _ in range(len(voting_rules))]
    manipulable = [[] for _ in range(len(voting_rules))]
    for _ in tqdm(range(num_tests)):
        # Check average ranks of winner
        num_voters = np.random.choice(np.arange(80, 150))
        num_candidates = np.random.choice(np.arange(10, 80))
        profiles = generate_random_profiles(num_voters, num_candidates)

        for idx, rule in enumerate(voting_rules):
            winner = find_winner(profiles, rule)
            avg_rank = find_winner_average_rank(profiles, winner)
            average_ranks[idx].append(avg_rank / num_candidates)

        # Check if profile is manipulable or not
        num_voters = np.random.choice(np.arange(7, 11))
        num_candidates = np.random.choice(np.arange(3, 7))
        profiles = generate_random_profiles(num_voters, num_candidates)
        
        for idx, rule in enumerate(voting_rules):
            manipulable[idx].append(check_manipulable(profiles, rule, find_winner))


    # Plot average ranks as a histogram
    for idx, rule in enumerate(voting_rules):
        plt.hist(average_ranks[idx], alpha=0.8, label=rule)

    plt.legend()
    plt.xlabel('Fractional average rank of winner')
    plt.ylabel('Frequency')
    plt.savefig('average_ranks.jpg')
    
    # Plot bar chart for fraction of manipulable profiles
    manipulable = np.sum(np.array(manipulable), axis=1)
    manipulable = np.divide(manipulable, num_tests)
    plt.clf()
    plt.bar(voting_rules, manipulable)
    plt.ylabel('Manipulability fraction')
    plt.savefig('manipulable.jpg')