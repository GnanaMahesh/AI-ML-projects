#!/usr/bin/python
import sys
import numpy as np
import itertools

def intersect(a, b):
    """
    Finds the intersection of two lists.
    Args:
    a (list): The first list.
    b (list): The second list.
    Returns:
    list: A list containing the elements common to both input lists.
    """
    return list(set(a) & set(b))

def make_util_matrix(num_players_info, num_strategies_per_player, util_list):
    """
    Constructs a utility matrix based on the number of players, their strategies, and a list of utilities.
    Args:
    num_players_info (int): The number of players.
    num_strategies_per_player (list): A list containing the number of strategies each player has.
    util_list (list): A list containing utilities for each combination of strategies.
    Returns:
    numpy.ndarray: A utility matrix representing the game.
    """
    tup_list = []
    for i in range(0, len(util_list), num_players_info):
        temp_list =  []
        for j in range(num_players_info):
            temp_list.append(util_list[i+j])
        tup_list.append(tuple(temp_list))  
    temp_str = "float"
    temp_str2 = ",float"*(num_players_info-1)
    string = temp_str+temp_str2
    dt = np.dtype(string)
    data = np.array(tup_list, dtype=dt)
    tup = tuple(x for x in num_strategies_per_player)
    util_matrix = data.reshape(tup[::-1])
    util_matrix = np.transpose(util_matrix)
    return util_matrix

def make_allpermut(num_players_info, num_strategies_per_player):
    """
    Args:
    num_players_info (int): The number of players.
    num_strategies_per_player (list): A list containing the number of strategies each player has.
    Returns:
    list: A list of all possible permutations of strategies.
    """
    somelists=[]
    temp=[[]  for i in range(num_players_info)] 
    for k in range(num_players_info):
        temp[k]=[i for i in range(num_strategies_per_player[k])]
        somelists.append(temp[k])

    allpermut=list(itertools.product(*somelists))
    for i in range(len(allpermut)):
        allpermut[i]=list(allpermut[i])
    return allpermut

def check(array, target, forstrat, num_players_info, num_strategies_per_player, util_matrix):
    #get values from allpermut, check for comparison in target, if this val itself is max return 1 ,else return 0
    """
    Checks if a given strategy is a best response to a target strategy for all players.
    Args:
    array (list): A list of indices representing strategies to be compared.
    target (int): The index of the target strategy.
    forstrat (list): The current strategy being evaluated.
    num_players_info (int): The number of players.
    num_strategies_per_player (list): A list containing the number of strategies each player has.
    util_matrix (numpy.ndarray): The utility matrix representing the game.
    Returns:
    int: 1 if the given strategy is a best response, otherwise 0.
    """
    allpermut=make_allpermut(num_players_info, num_strategies_per_player)
    tf=0
    getindex=[]
    for i in range(len(array)):
        getindex.append(allpermut[array[i]])
    for i in range(len(getindex)):
        if util_matrix[tuple(getindex[i])][target]<=util_matrix[tuple(forstrat)][target]:
            tf+=1
    if tf==len(array):        
        return 1
    else:
        return 0

def sel_index(player, args, multiplier, num_players_info):
    """
    Selects the index in the game data based on the player and their chosen strategies.
    Args:
    player (int): The player for whom the index is calculated.
    args (list): A list containing the chosen strategies of all players.
    multiplier (list): Cumulative products of the list with number of strategies.
    num_players_info (int): The number of players.
    Returns:
    int: The calculated index in the game data.
    """
    result = 0
    i = 0
    for arg in args:
        result = result + (arg * multiplier[i])
        i = i + 1
    result = result * num_players_info
    result += player
    return result

def find_strongly_dominant_eq(gamedata, playerno, totalplayer, topplayer, num_strategies_per_player, 
                              multiplier, num_players_info, strategyarr = [], eqindex = -1):
    """
    Finds strongly dominant strategy equilibrium for a given player.
    Args:
    gamedata (list): The payoff matrix in Gambit nfg format 
    playerno (int): The index of the player for whom the equilibrium strategy is sought.
    totalplayer (list): A list of remaining players to consider.
    topplayer (int): The index of the top player in the list of remaining players.
    num_strategies_per_player (list): List containing the number of strategies of each player.
    multiplier (list): Cumulative product of the "strategies" list, used to find the index of
                       a player's utility within a strategy (using the sel_index function).
    num_players_info (int): The number of players.    
    strategyarr (list, optional): The list of strategies chosen so far. Defaults to [].
    eqindex (int, optional): The index of the equilibrium strategy found so far. Defaults to -1.
    Returns:
    int: The index of the equilibrium strategy or -sys.maxsize if none exists.
    """
    if len(totalplayer) >= 1:
        cur_player = totalplayer[0]
        temp = 0
        totalplayer = totalplayer[1:]
        for strategy in range(num_strategies_per_player[cur_player]):
            temparray = strategyarr[:]
            temparray.append(strategy)
            temp = find_strongly_dominant_eq(gamedata, playerno, totalplayer, topplayer, num_strategies_per_player, 
                                             multiplier, num_players_info, temparray, eqindex)
            if temp == -sys.maxsize:
                return temp
            else:
                eqindex = temp
        return temp
    

    else:
        max_payoff = -sys.maxsize
        max_index = -1
        other_payoffs = []
        for i in range(num_strategies_per_player[playerno]):
            index = sel_index(playerno, strategyarr[:playerno]+[i]+strategyarr[(playerno+1):], multiplier, num_players_info)
            payoff = gamedata[index]
            if max_payoff < payoff:
                max_payoff = payoff
                max_index = i
            else:
                other_payoffs.append(payoff)

        if max_payoff in other_payoffs:
            return -sys.maxsize
        if eqindex == -1:
            eqindex = max_index
        elif eqindex != max_index:
            return -sys.maxsize
        return eqindex

def find_weakly_dominant_eq(gamedata, playerno, totalplayer, topplayer, num_strategies_per_player, 
                            multiplier, num_players_info, eqindex, strategyarr = []):
    """
    Finds weakly dominant equilibrium strategy for a given player.
    Args:
    gamedata (list): The payoff matrix in Gambit nfg format 
    playerno (int): The index of the player for whom the equilibrium strategy is sought.
    totalplayer (list): A list of remaining players to consider.
    topplayer (int): The index of the top player in the list of remaining players.
    num_strategies_per_player (list): List containing the number of strategies of each player.
    multiplier (list): Cumulative product of the "strategies" list, used to find the index of
                       a player's utility within a strategy (using the sel_index function).
    num_players_info (int): The number of players.
    eqindex (int or list): The index/es of the equilibrium strategy found so far.
    strategyarr (list, optional): The list of strategies chosen so far. Defaults to [].
    Returns:
    tuple: The index of the equilibrium strategy and the updated equilibrium index list.
    """
    if len(totalplayer) >= 1:
        cur_player = totalplayer[0]
        temp = 0
        totalplayer = totalplayer[1:]
        for strategy in range(num_strategies_per_player[cur_player]):
            temparray = strategyarr[:]
            temparray.append(strategy)
            temp, eqindex = find_weakly_dominant_eq(gamedata, playerno, totalplayer, topplayer, num_strategies_per_player,
                                                    multiplier, num_players_info, eqindex, temparray)
            if temp == -sys.maxsize:
                return temp, eqindex
        return temp, eqindex
    else:
        max_payoff = -sys.maxsize
        max_index = []
        for i in range(num_strategies_per_player[playerno]):
            index = sel_index(playerno, strategyarr[:playerno]+[i]+strategyarr[(playerno+1):], multiplier, num_players_info)
            payoff = gamedata[index]
            if max_payoff < payoff:
                max_payoff = payoff
                max_index = []
                max_index.append(i)
            elif max_payoff == payoff:
                max_index.append(i)
        if eqindex[0] == -1:
            eqindex = max_index
        else:
            temp = intersect(eqindex,max_index)
            eqindex = temp[:]
        if not eqindex:
            return -sys.maxsize, eqindex
        else:
            return eqindex[0], eqindex

def psne_gen(num_players_info, num_strategies_per_player, util_matrix):
    """
    Finds Pure Nash equilibrium strategies.
    Args:
    num_players_info (int): The number of players.
    num_strategies_per_player (list): A list containing the number of strategies each player has.
    util_matrix (numpy.ndarray): The utility matrix representing the game.
    Returns:
    list: A list of Pure Strategy Nash Equilibriums.
    """
    psne_list=[]
    stratpossibilities = make_allpermut(num_players_info, num_strategies_per_player)
    for possibility in stratpossibilities:
      ispsne = True
      for i in range(num_players_info):
          for j in range(num_strategies_per_player[i]):
              if(util_matrix[tuple(possibility)][i] < util_matrix[tuple(possibility[:i]+[j]+possibility[i+1:])][i]):
                ispsne = False
                break   
      if ispsne == True:
          psne_list.append(possibility)
    return psne_list 

def msne_gen(num_players_info, num_strategies_per_player, util_matrix):
    """
    Finds Mixed Nash equilibrium strategies.
    Args:
    num_players_info (int): The number of players.
    num_strategies_per_player (list): A list containing the number of strategies each player has.
    util_matrix (numpy.ndarray): The utility matrix representing the game.
    Returns:
    list: The MSNE in the form [[p_1*, p_2*], [q_1*, q_2*]].
    """
    msne=[]
    u_111 = util_matrix[0][0][0]
    u_112 = util_matrix[0][0][1]
    u_121 = util_matrix[0][1][0]
    u_211 = util_matrix[1][0][0]
    u_122 = util_matrix[0][1][1]
    u_212 = util_matrix[1][0][1]
    u_221 = util_matrix[1][1][0]
    u_222 = util_matrix[1][1][1]
    p1 = (u_221 - u_211)/(u_111 - u_211 + u_221 - u_121)
    q1 = (u_222 - u_212)/(u_112 - u_212 + u_222 - u_122)

    if p1 > 0 and p1 < 1 and q1 > 0 and q1 < 1:
        msne = [[q1,1-q1],[p1,1-p1]]
    return msne

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please pass the name of the game file to be analyzed")
    f = open(sys.argv[1], "r")
    num_players_info_line = f.readline()
    data = f.readline().split(" ")
    data = data[data.index("{") + 1: data.index("}\n")]
    data = data[data.index("{") + 1:]
    num_players_info = len(data)
    num_strategies_per_player = list(map(int, data))
    multiplier = []
    temp = 1
    for i in range(len(num_strategies_per_player)):
        multiplier.append(temp)
        temp = temp * num_strategies_per_player[i]

    f.readline()
    data = f.readline().split(" ")
    gamedata = list(map(int, data))
    
    ###############     Equilibrium     ###############
    playerslist = list(range(num_players_info))
    return_value = -1
    strong_eq = []
    for i in range(num_players_info):
        tempplayerlist = playerslist[:]
        tempplayerlist.remove(i)
        value = find_strongly_dominant_eq(gamedata, i, tempplayerlist, tempplayerlist[0], num_strategies_per_player, multiplier, num_players_info)
        if value == -sys.maxsize:
            print("No Strongly Dominant Strategy Equilibrium exists\n")
            return_value = 0
            break
        else:
            strong_eq.append(value)
    if return_value == -1:
        print(f"Strongly Dominant Strategy Equilibrium (in order of P1, P2, ... , Pn) is: {strong_eq}\n")
    else:
        min_eq_list = []
        for i in range(num_players_info):
            tempplayerlist = playerslist[:]
            tempplayerlist.remove(i)
            result_index = [-1]
            value, result_index = find_weakly_dominant_eq(gamedata, i, tempplayerlist, tempplayerlist[0], num_strategies_per_player, multiplier, num_players_info, result_index)
            if value == -sys.maxsize or len(result_index) == num_strategies_per_player[i]:
                print("No Weakly Dominant Strategy Equilibrium exists as well\n")
                return_value = -2
                break
            else:
                min_eq_list.append(result_index)

        if return_value != -2:
            print(f"Weakly Dominant Strategy Equilibrium(s) is (are): {min_eq_list}\n")

    util_matrix = make_util_matrix(num_players_info, num_strategies_per_player, gamedata)
    psnelist = psne_gen(num_players_info, num_strategies_per_player, util_matrix)
    if len(psnelist) == 0:
        print("No Pure Strategy Nash Equilibrium exists either")
    else:
        print(f"PSNEs: {psnelist}")

    if len(psnelist) % 2 == 0 and num_players_info == 2 and num_strategies_per_player[0] == 2 and num_strategies_per_player[1] == 2:
        msne = msne_gen(num_players_info, num_strategies_per_player, util_matrix)
        print(f"\nMSNE: {msne}")
