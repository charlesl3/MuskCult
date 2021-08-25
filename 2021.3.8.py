#!/usr/bin/env python3
\
# -*- coding: utf-8 -*-

from pyomo.environ import *
from pyomo.opt import *
from pyomo.core import *
import numpy as np
from math import ceil
import pandas as pd
import os
from datetime import date, timedelta,datetime
import warnings
warnings.filterwarnings("ignore")
from textwrap import wrap
import easy_date 
import date_converter
import re
import math
import matplotlib.pyplot as plt
import statistics 
#%% Sets

# Set of possible trip
possible_trip = pd.read_csv('/Users/charlesliu/Desktop/PhD/CRO/real data/POSSIBLETRIPS.csv')
possibletrip = []
for ind in possible_trip.index:
     possibletrip.append([str(possible_trip['TripId'][ind]),str(possible_trip['OnwardTripId'][ind])])
possible1 = [str(i) for i in set(possible_trip['TripId'])] # trips in the "possibletrip" data

preassignment_1 = pd.read_csv('/Users/charlesliu/Desktop/PhD/CRO/real data/PREASSIGNMENTDATA.csv')
start_con = pd.to_datetime(preassignment_1['DutyStart'], format='%Y-%m-%dT%H:%M:%S')>pd.Timestamp('2020-06-30 00:00:00')
end_con = pd.to_datetime(preassignment_1['DutyEnd'], format='%Y-%m-%dT%H:%M:%S')>pd.Timestamp('2020-07-01 00:00:00')
end2_con = pd.to_datetime(preassignment_1['NextAvailable'], format='%Y-%m-%dT%H:%M:%S')>pd.Timestamp('2020-07-01 00:00:00')

preassignment_2 = preassignment_1[start_con]
preassignment_3 = preassignment_2[end_con]
preassignment = preassignment_3[end2_con]
preassignment = preassignment[preassignment['DutyType']==16]
pre = [str(i) for i in list(preassignment['DutyID'])]
pre_crew = [str(i) for i in list(preassignment['CrewID'])]
pre_act = list(zip(pre_crew,pre))     
pre_trip_df = preassignment[preassignment['TripId'].notna()]
pre_trip = [str(i) for i in list(pre_trip_df['DutyID'])]

pre_duty_intrip = [('1044','3084','3085'),('1125','3847','3848'),('1223','4134','4135'),('49376','5530','5531')]

# Set of valid crew and trip
valid_crew_trip = pd.read_csv('/Users/charlesliu/Desktop/PhD/CRO/real data/VALIDCREWTRIP.csv')
validcrewtrip = []
for ind in valid_crew_trip.index:  
     validcrewtrip.append((str(valid_crew_trip['CrewId'][ind]),str(valid_crew_trip['TripId'][ind])))
D = validcrewtrip + pre_act

assignable_crew = dict(zip(list(set(valid_crew_trip['CrewId'])),[()]*len(set(valid_crew_trip['CrewId']))))
assignable_crew = {str(u):v for u,v in assignable_crew.items()}
for i in validcrewtrip:
    assignable_crew[i[0]] = assignable_crew[i[0]]+(i[1],)

# set of crew C
Crew_data = pd.read_csv('/Users/charlesliu/Desktop/PhD/CRO/real data/CREW.csv')
C_total = [str(i) for i in Crew_data['CrewID']]
C = [str(i) for i in set(valid_crew_trip['CrewId'])]

pre_all_poss = [(i,j) for i in C for j in pre]
# set of trips P
trip_data = pd.read_csv('/Users/charlesliu/Desktop/PhD/CRO/real data/TRIPS.csv')
P = [str(i) for i in set(trip_data['TripId'])]
all_act = pre + P

'''
# cost, in Laminaar all zeros
c_ij = {}
for i in C:
    for j in P:
        if [i, j] in validcrewtrip:
            c_ij.update({
                (i, j):
                valid_crew_trip.loc[(valid_crew_trip.CrewId == int(i)) & (valid_crew_trip.TripId == int(j)), 'CROCrewTieBreakerWeightage'].values[0]})
        else:
            c_ij.update({(i, j): 0})
'''


# set of flights F
flight_data = pd.read_csv('/Users/charlesliu/Desktop/PhD/CRO/real data/FLIGHTDTLS.csv')
F = [str(i) for i in set(flight_data['FlightId'])]


# set of Actype Type
Type = ['2','4','5']


# set of Patterns Pt
Pt = ['525HYDCCU/743CCUIXA/744IXACCU', '541HYDDEL/839DELHYD', '546HYDMAA', '559HYDDEL/542DELHYD/542HYDTIR/541TIRHYD',
     '620HYDBOM/625BOMLKO/626LKOBOM', '952HYDVTZ/452VTZDEL', '977HYDMCT/978MCTHYD'] # patterns in only obj functions, this is what we are interested in
Pt_total = [str(i) for i in set(trip_data['TripPattern'])] # patterns in trips only 
Pt_all = list(set(list(set(preassignment['TripPattern']))[1:]+Pt_total)) #include patterns unique in preassignments



# set of diurnal DI
DI = ['360','1081']
unique_trip = trip_data.drop_duplicates().dropna()
groups = unique_trip[['TripId','AircraftID','NoOfLandings','DiurnalStart','InternationalCount','FDTLGroupID']].drop_duplicates().dropna()
group1 = groups[groups['FDTLGroupID']==8656].reset_index()
group2 = groups[groups['FDTLGroupID']==8658].reset_index() # not the same for 'AircraftID','NoOfLandings'

#%% consecutively assignable trips
triptime_unique_trip = unique_trip[['TripId','PreRestStartTimeBeforeFirstDuty','RestEndTimeOfLastDuty']].drop_duplicates()
triptime_preassignment = preassignment[['DutyID','DutyStart','NextAvailable']]
triptime_all_act = pd.DataFrame( np.concatenate( (triptime_unique_trip.values, triptime_preassignment.values), axis=0 ))
triptime_all_act.columns = [ 'actID', 'start', 'end' ]
triptime_all_act['dt_start'] = pd.to_datetime(triptime_all_act['start'], format='%Y-%m-%dT%H:%M:%S')
triptime_all_act['dt_end'] = pd.to_datetime(triptime_all_act['end'], format='%Y-%m-%dT%H:%M:%S')


triptime_unique_trip_alone = unique_trip[['TripId','PreRestStartTimeBeforeFirstDuty','TripEndTime']].drop_duplicates() # alone means no postrest, no NextAvailable
triptime_preassignment_alone = preassignment[['DutyID','DutyStart','DutyEnd']]
triptime_all_act_alone = pd.DataFrame( np.concatenate( (triptime_unique_trip_alone.values, triptime_preassignment_alone.values), axis=0 ))
triptime_all_act_alone.columns = [ 'actID', 'start', 'end' ]
triptime_all_act_alone['dt_start'] = pd.to_datetime(triptime_all_act_alone['start'], format='%Y-%m-%dT%H:%M:%S')
triptime_all_act_alone['dt_end'] = pd.to_datetime(triptime_all_act_alone['end'], format='%Y-%m-%dT%H:%M:%S')


g_j = {} #  duration of a trip, in hours
for j in all_act:
    start = triptime_all_act_alone[triptime_all_act_alone['actID'] == int(j)]['dt_start'].values[0]
    end = triptime_all_act_alone[triptime_all_act_alone['actID'] == int(j)]['dt_end'].values[0]
    delta = (end-start).astype('timedelta64[s]').astype(int)
    delta = delta /3600
    g_j.update({j:delta})
       
    
Q1 = []          
for j in set(triptime_all_act['actID']):
    end = triptime_all_act[triptime_all_act['actID'] == int(j)]['dt_end'].values[0]
    new_data = triptime_all_act[triptime_all_act['dt_start'] > end]
    for j2 in set(new_data['actID']):
        Q1.append((str(j),str(j2)))

#Q2 = Q1+[(j[1],j[0]) for j in Q1]      
# Q:(i,j,j') consecutively assignable activities
Q = []
for i in C:
    for j in Q1:
        if ((i,j[0]) in D) and ((i,j[1]) in D):
            Q.append((i,j[0],j[1]))

# tau_ijj_prime: end-start interval to each crew member of their available activities
tau_ijj_prime = {}
for i in Q:
    start = triptime_all_act_alone[triptime_all_act_alone['actID'] == int(i[2])]['dt_start'].values[0]
    end = triptime_all_act_alone[triptime_all_act_alone['actID'] == int(i[1])]['dt_end'].values[0]
    delta = (start-end).astype('timedelta64[s]')/np.timedelta64(3600, 's')
    tau_ijj_prime.update({i:delta})

# r_ijj_prime: whether there could be a weekly rest, 36 hour and base 
r_ijj_prime = tau_ijj_prime.copy()
for u,v in r_ijj_prime.items():
    if v >= 36:
        r_ijj_prime[u] = 1
    else:
        r_ijj_prime[u] = 0

for u,v in r_ijj_prime.items():
    if (u[1] in pre) and (u[2] not in pre):
        base1 = preassignment[preassignment['DutyID']== int(u[1])]['EndBase'].values[0]
        if base1 != 'HYD ':
            r_ijj_prime[u] = 0
    elif (u[1] not in pre) and (u[2] in pre):
        base2 = preassignment[preassignment['DutyID']== int(u[2])]['StartBase'].values[0]
        if base2 != 'HYD ':
            r_ijj_prime[u] = 0
    elif (u[1] in pre) and (u[2] in pre):
        base3 = preassignment[preassignment['DutyID']== int(u[2])]['StartBase'].values[0]
        base4 = preassignment[preassignment['DutyID']== int(u[1])]['EndBase'].values[0]
        if (base3!= 'HYD ') or (base4!= 'HYD ') or (base3!=base4):
            r_ijj_prime[u] = 0

for u,v in r_ijj_prime.items():
    if u in pre_duty_intrip:
         r_ijj_prime[u] = 0
        
        


tau1_ijj_prime = tau_ijj_prime.copy() 
for u,v in tau1_ijj_prime.items():
    if r_ijj_prime[u] == 1:
        tau1_ijj_prime[u] = 0

tau2_ijj_prime =  tau_ijj_prime.copy()

first_trip = [] # this is the first-only trip (crew, trip) pair
for i in C:
    list_pair_1 = [q[1] for q in Q if q[0] == i]
    list_pair_2 = [q[2] for q in Q if q[0] == i]
    for j in all_act:
        if (j in list_pair_1) and (j not in list_pair_2):
            first_trip.append((i,j))
                   
#%% flight duties inside a trip

trip_duty_data = unique_trip[['TripId','DutyId','DutyStartTime','DutyEndTime']].drop_duplicates().dropna()
 
flight_duty = list(map(str,set(trip_duty_data['DutyId'])))

trip_duty_dict = {(j,d):0 for j in P for d in flight_duty}       

for ind in trip_duty_data.index:
    trip1 = str(trip_duty_data['TripId'][ind])
    duty1 = str(trip_duty_data['DutyId'][ind])
    trip_duty_dict[(trip1,duty1)] = 1
    
duty_in_trip = dict(zip(P, [()]*len(P))) 
for i in list(trip_duty_dict.keys()):
    if trip_duty_dict[i]==1:
        duty_in_trip[i[0]] = duty_in_trip[i[0]]+(i[1],)

   
#%% flight/trip inclusion

# flying time of flight f, t_f in hours
unique_flight = flight_data.drop_duplicates()
unique_flight['STA'] =  pd.to_datetime(unique_flight['STA'], format='%Y-%m-%d'+'T'+'%H:%M:%S')
unique_flight['STD'] =  pd.to_datetime(unique_flight['STD'], format='%Y-%m-%d'+'T'+'%H:%M:%S')
unique_flight['flight time'] = unique_flight['STA']-unique_flight['STD']
unique_flight['flight time'] = unique_flight['flight time'] / np.timedelta64(1, 'm') # in hour
t_f = {}
for f in unique_flight['FlightId']:
      t_f.update({str(f):unique_flight.loc[unique_flight.FlightId == f, 'flight time'].values[0]/60}) 
      

# flying time of flight j, t_j, in hours
unique_trip = trip_data.drop_duplicates()
t_j = {}
for j in unique_trip['TripId']:
    fly_min = sum(set(unique_trip[unique_trip['TripId'] == int(j)]['FlyingHrs']))
    t_j.update({str(j):fly_min/60})    


# alpha_fj, check flight f is in trip j
alpha_fj = {}
unique_trip = unique_trip.dropna()
for f in list(t_f.keys()):
    for j in list(t_j.keys()):
        alpha_fj.update({(f,j):0})
        sub_data = unique_trip.loc[unique_trip.TripId == int(j)] 
        for ind in sub_data.index:
            if f in [sub_data['FlightIds'][ind][i:i+7] for i in range(0, len(sub_data['FlightIds'][ind]),7)]:
                alpha_fj.update({(f,j):1})


# trip_flight: which trip contains which flights
trip_flight = dict(zip(P, [()]*len(P))) 
for i in list(alpha_fj.keys()):
    if alpha_fj[i]==1:
        trip_flight[i[1]] = trip_flight[i[1]]+(i[0],)


#%% time window related
# rolling time window generator
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)       
def days_ago_date(date, days):
    date2 = date - timedelta(days = days)
    return date2
def get_rolling_time_window(rostering_start_date,rostering_end_date,timewindow):
    days_between = (rostering_end_date-rostering_start_date).days
    date_ago = days_ago_date(rostering_start_date,timewindow-1)
    first_window = [dt for dt in daterange(date_ago,rostering_start_date)]
    next_window = first_window
    rolling_window = [first_window]
    
    for i in range(days_between):
        rolling_window.append([dt+timedelta(days=1) for dt in next_window])
        next_window = [dt+timedelta(days=1) for dt in next_window]
    rolling_window = [tuple(i) for i in rolling_window]
               
    return rolling_window



# delta_fn, check if flight f is in n(time window)
STD_dates = {}
STA_dates = {}
for ind in unique_flight.index: 
     STD_dates.update({str(unique_flight['FlightId'][ind]):unique_flight['STD'][ind].date()})
for ind in unique_flight.index: 
     STA_dates.update({str(unique_flight['FlightId'][ind]):unique_flight['STA'][ind].date()})
STD_dates == STA_dates #true, no over-day flights 
flight_dates = STA_dates

T7_window = get_rolling_time_window(date(2020,7,1),date(2020,7,31),7)

def get_delta_fn(time_window_list):
    delta_fn = {}
    for n in time_window_list:
        for f in set(flight_dates.keys()):
            if STA_dates[f] in n:
                delta_fn.update({tuple([f,n]):1})
            else:
                delta_fn.update({tuple([f,n]):0})
    return delta_fn 
get_delta_fn(T7_window)




# delta_jn, check if j is in window n
trip_Dep_dates = {}
trip_Arr_dates = {}

for j in trip_duty_data['DutyId']:
    trip_Dep_dates.update({
        str(j):
        datetime.strptime(
            trip_duty_data.loc[trip_duty_data.DutyId == j, 'DutyStartTime'].values[0]
            [0:10], '%Y-%m-%d').date()
    })
for j in trip_duty_data['DutyId']:
    trip_Arr_dates.update({
        str(j):
        datetime.strptime(
            trip_duty_data.loc[trip_duty_data.DutyId == j, 'DutyEndTime'].values[0]
            [0:10], '%Y-%m-%d').date()
    })

def get_delta_dn(time_window_list):
    delta_jn = {}
    for n in time_window_list:
        for j in set(trip_Dep_dates.keys()):
            if trip_Dep_dates[j] in n or trip_Arr_dates[j] in n:
                delta_jn.update({tuple([j, n]): 1})
            else:
                delta_jn.update({tuple([j, n]): 0})
    return delta_jn
get_delta_dn(T7_window)



# deal with overlapped timewindow for j, in hours
#triptime_unique_data = unique_trip[['TripId','PreRestStartTimeBeforeFirstDuty','RestEndTimeOfLastDuty','TripEndTime']].drop_duplicates().reset_index()
def duty_hour_within_window(rollingtimewindow):
    tj = {}
    for ind in trip_duty_data.index:
        for n in rollingtimewindow:

            if datetime.strptime(trip_duty_data['DutyStartTime'][ind], '%Y-%m-%dT%H:%M:%S') > date_converter.date_to_datetime(
                        n[0]) and datetime.strptime(trip_duty_data['DutyEndTime'][ind], '%Y-%m-%dT%H:%M:%S') < date_converter.date_to_datetime(n[len(n) - 1]):
                interval_1 = datetime.strptime(trip_duty_data['DutyEndTime'][ind], '%Y-%m-%dT%H:%M:%S') - datetime.strptime(trip_duty_data['DutyStartTime'][ind], '%Y-%m-%dT%H:%M:%S')
                hours_1 = interval_1.days*24+interval_1.seconds/3600
                tj.update({(str(trip_duty_data['DutyId'][ind]),n):hours_1})
            
            elif datetime.strptime(trip_duty_data['DutyStartTime'][ind], '%Y-%m-%dT%H:%M:%S') < date_converter.date_to_datetime(
                        n[0]) and datetime.strptime(trip_duty_data['DutyEndTime'][ind], '%Y-%m-%dT%H:%M:%S') > date_converter.date_to_datetime(n[0]) and datetime.strptime(trip_duty_data['DutyEndTime'][ind], '%Y-%m-%dT%H:%M:%S') < date_converter.date_to_datetime(n[len(n) - 1]):     
                interval_2 = datetime.strptime(trip_duty_data['DutyEndTime'][ind], '%Y-%m-%dT%H:%M:%S') - date_converter.date_to_datetime(n[0])
                hours_2 = interval_2.days*24+interval_2.seconds/3600
                tj.update({(str(trip_duty_data['DutyId'][ind]),n):hours_2})
            
            elif datetime.strptime(trip_duty_data['DutyStartTime'][ind], '%Y-%m-%dT%H:%M:%S') > date_converter.date_to_datetime(
                        n[0]) and datetime.strptime(trip_duty_data['DutyStartTime'][ind], '%Y-%m-%dT%H:%M:%S') < date_converter.date_to_datetime(n[len(n)-1]) and datetime.strptime(trip_duty_data['DutyEndTime'][ind], '%Y-%m-%dT%H:%M:%S') > date_converter.date_to_datetime(n[len(n) - 1]):
                interval_3 = date_converter.date_to_datetime(n[len(n) - 1]) - datetime.strptime(trip_duty_data['DutyStartTime'][ind], '%Y-%m-%dT%H:%M:%S')
                hours_3 = interval_3.days*24+interval_3.seconds/3600
                tj.update({(str(trip_duty_data['DutyId'][ind]),n):hours_3})
            
            elif datetime.strptime(trip_duty_data['DutyStartTime'][ind], '%Y-%m-%dT%H:%M:%S') < date_converter.date_to_datetime(
                        n[0]) and datetime.strptime(trip_duty_data['DutyEndTime'][ind], '%Y-%m-%dT%H:%M:%S') > date_converter.date_to_datetime(n[len(n) - 1]):
                interval_4 = date_converter.date_to_datetime(n[len(n) - 1]) - date_converter.date_to_datetime(n[0])
                hours_4 = interval_4.days*24+interval_4.seconds/3600
                tj.update({(str(trip_duty_data['DutyId'][ind]),n):hours_4})
                
            else:
                tj.update({(str(trip_duty_data['DutyId'][ind]),n):0})             
                
    return tj

duty_hour_within_window(T7_window)


#%% overlap trips

# gamma_j1j2: overlapped trip issue 
triptime_unique_data = unique_trip[['TripId','PreRestStartTimeBeforeFirstDuty','RestEndTimeOfLastDuty','TripEndTime']].drop_duplicates().reset_index()
gamma_j1j2 = {}
for ind1 in triptime_unique_data.index:
    for ind2 in triptime_unique_data.index:
        trip1_start = datetime.strptime(triptime_unique_data['PreRestStartTimeBeforeFirstDuty'][ind1], '%Y-%m-%dT%H:%M:%S')
        trip1_end = datetime.strptime(triptime_unique_data['RestEndTimeOfLastDuty'][ind1], '%Y-%m-%dT%H:%M:%S')
        trip2_start = datetime.strptime(triptime_unique_data['PreRestStartTimeBeforeFirstDuty'][ind2], '%Y-%m-%dT%H:%M:%S')
        trip2_end = datetime.strptime(triptime_unique_data['RestEndTimeOfLastDuty'][ind2], '%Y-%m-%dT%H:%M:%S')
        if  (trip1_end < trip2_start) or (trip1_start > trip2_end):
            gamma_j1j2.update({(str(triptime_unique_data['TripId'][ind1]),str(triptime_unique_data['TripId'][ind2])):0})
        else:
            gamma_j1j2.update({(str(triptime_unique_data['TripId'][ind1]),str(triptime_unique_data['TripId'][ind2])):1})
#%% trip and preassignment patterns


# which trip has which pattern: note that the trip only cares about the first pattern
# this is the patter of the trips
trip_pattern = {}
for j in unique_trip['TripId']:
    pattern_sort = unique_trip[unique_trip['TripId'] == int(j)].sort_values(by=['DutyStartTime'], ascending=True)
    trip_pattern.update({str(j):pattern_sort['TripPattern'].iloc[0]})



# this is the pattern of pre-assigned activities, no-flying to be zero
pre_pattern = {}
for j in pre:
    if j in pre_trip:
        pt = preassignment[preassignment['DutyID'] == int(j)]['TripPattern'].values[0]
        pre_pattern.update({j:pt})
    else:
        pre_pattern.update({j:0})


pre_pattern_number = {} # binary
for i in C:
    for p in Pt_all:
        pre_pattern_number.update({(i,p):0})
        if int(i) in set(preassignment['CrewID']) and p in preassignment[preassignment['CrewID'] == int(i)]['TripPattern'].value_counts():
            pre_pattern_number.update({(i,p):preassignment[preassignment['CrewID'] == int(i)]['TripPattern'].value_counts()[p]})

# binary parameters to indicate which trip/pre has which pattern(s)    
p1_jp = {} # here only use selected patterns, not all patterns
for j in P:
    for p in Pt:
        if p in trip_pattern[str(j)]:
            p1_jp.update({(str(j),p):1})
        else:
            p1_jp.update({(str(j),p):0})

p2_jp = {}
for j in pre:
    for p in Pt:
        if p == pre_pattern[str(j)]:
            p2_jp.update({(str(j),p):1})
        else:
            p2_jp.update({(str(j),p):0})

# binary parameter of patterns, all activities
p_jp = {**p1_jp, **p2_jp}

#%% other distribution para

diur_trip = unique_trip[['TripId','DiurnalStart']].drop_duplicates() # some trips have two Diurnal types
# diur_jd
diur_jd = dict(zip(list(zip(P,['360']*len(P)))+list(zip(P,['1081']*len(P))),[0]*2*len(P)))
for ind in diur_trip.index:
    trip = str(diur_trip['TripId'][ind])
    diur = str(diur_trip['DiurnalStart'][ind])
    diur_jd.update({(trip,diur):1})
diur_jd[('1732627','1081')]
diur_jd[('1732627','360')]

# Ty_jt whether j is of type t

Ty_trip = unique_trip[['TripId','AircraftID']].drop_duplicates() # some trips have multiple AC types
Ty_list = [(j,ty) for j in P for ty in ['0','2','4','5']]
Ty_jt = dict(zip(Ty_list,[0]*len(Ty_list)))
for ind in Ty_trip.index:
    trip = str(Ty_trip['TripId'][ind])
    Ty = str(Ty_trip['AircraftID'][ind])
    Ty_jt.update({(trip,Ty):1})
Ty_jt[('1731418','5')]
Ty_jt[('1731418','2')]

# layover_j how many layovers, # of duties - 1 
trip_flightduty = trip_data[['TripId','DutyId']].drop_duplicates()
layover_data = trip_flightduty.groupby(['TripId']).size().reset_index(name='layover')
layover_j = {str(layover_data['TripId'][ind]):layover_data['layover'][ind]-1 for ind in layover_data.index}


# penalty_j: penalty for not covering a trip j
penalty_trip = unique_trip[['TripId','UncoveredPairingSlackCost']].drop_duplicates()
penalty_j = {}
for ind in penalty_trip.index:
    trip = str(penalty_trip['TripId'][ind])
    penalty = penalty_trip['UncoveredPairingSlackCost'][ind]
    penalty_j.update({trip:penalty})

# I_j international
Int_trip = unique_trip[['TripId','FDTLGroupID','InternationalCount']].drop_duplicates()
#Int_trip = Int_trip[Int_trip['FDTLGroupID']==8658]
I_j = dict(zip(P,[0]*len(P)))
# for ind in Int_trip.index:
#     trip = str(Int_trip['TripId'][ind])
#     intcount = Int_trip['InternationalCount'][ind]
#     I_j.update({trip:intcount})   

for j in P:
    int_count = set(unique_trip[unique_trip['TripId'] == int(j)]['InternationalCount'])
    if 1 in int_count:
       I_j[j]=1
     
I_j['1731371']

# ld_j landings
Land_trip = unique_trip[['TripId','NoOfLandings']].drop_duplicates()
#Land_trip = Land_trip[Land_trip['FDTLGroupID']==8658]
ld_j = {}
for j in P:
    nooflandings = Land_trip[Land_trip['TripId'] == int(j)]['NoOfLandings'].sum()
    ld_j.update({j:nooflandings})  
ld_j['1731418']

#%% values and past values

# weights and target value
w_BTPast = 0.00000667
t_BTPast = 3960

w_LastFlown = 0.0333
t_LastFlown = dict(zip(Pt,[28.4,30.1,30.5,31.5,31.5,29.1,34.8]))

w_NDF = 0.0286
t_NDF = dict(zip(Pt,[2.20,2.11,2.00,2.15,2.11,2.15,2.00]))

w_BT = 0.0000333
t_BT = 2430

w_other = 0.00000333
t_other = 2430

w_international = 0.1
t_international = 0.67

w_DI = 0.15
t_DI = dict(zip(DI,[8.13,8.43]))

w_AC = 0.0889
t_AC = dict(zip(Type,[2.25,6.53,6.37]))

w_Land = 0.167
t_Land = 36.5

w_Lay = 0.133
t_Lay = 3.1


# past values of BTPast, Int, Layover, Landing
pv_BTPast = {}
pv_International = {}
pv_Lay = {}
pv_Landing = {}

#for i in Crew_data['CrewID']:
for i in C:
    pv_BTPast.update({str(i):int(Crew_data.loc[Crew_data.CrewID == int(i), 'PastFlyingHours'].values[0]/60.0)})  
    pv_International.update({str(i):int(Crew_data.loc[Crew_data.CrewID == int(i), 'InternationalCount'].values[0])})
    pv_Lay.update({str(i):int(Crew_data.loc[Crew_data.CrewID == int(i), 'LayoverTripCount'].values[0])})
    pv_Landing.update({str(i):0})


# past values of patterns, all crew>30, all patterns>7
patterndata = pd.read_csv('/Users/charlesliu/Desktop/PhD/CRO/real data/TRIPPATTERNDATA.csv')
pv_pattern_crew = {}  # contains all patterns, not only the selected ones
for i in C:
    for p in Pt_all:
        pv_pattern_crew.update({(i, p): 0})

for ind in patterndata.index:
    pv_pattern_crew.update({(str(patterndata['CREWID'][ind]), patterndata['Routes'][ind]):patterndata['NoOfDays'][ind]})
pv_pattern_crew[('707', '050BOMHYD')]    

# past values of DI
DIdata = pd.read_csv('/Users/charlesliu/Desktop/PhD/CRO/real data/DIURNALDATA.csv')
pv_DI_crew = {}  # contains all patterns, not only the selected ones
for i in C:
    for di in DI:
        pv_DI_crew.update({(i, di): 0})
for ind in DIdata.index:
    pv_DI_crew.update({
        (str(DIdata['CrewId'][ind]), str(DIdata['DutyStartTime'][ind])):DIdata['CountOfDuties'][ind]})
pv_DI_crew[('598', '360')]

# past values of AC
ACdata = pd.read_csv('/Users/charlesliu/Desktop/PhD/CRO/real data/ACTYPEFLOWN.csv')
pv_AC_crew = {} # contains all patterns, not only the selected ones

for i in C:
    for t in Type:
        pv_AC_crew.update({(i,t):0})
           
for ind in ACdata.index:
        pv_AC_crew.update({(str(ACdata['CrewId'][ind]),str(ACdata['AcType'][ind])):ACdata['FlyingCount'][ind]})
pv_AC_crew[('770', '2')]


# past values of landings
Landingdata = pd.read_csv('/Users/charlesliu/Desktop/PhD/CRO/real data/NOOFLANDINGDUTY.csv')
for i in set(Landingdata['CrewId']):
    pv_Landing.update({
        str(i):
        (Landingdata.loc[Landingdata['CrewId'] == i]['DutyLandingCount'] *
         Landingdata.loc[Landingdata['CrewId'] == i]['CountOfDuties']).sum()
    })
pv_Landing['770']

#%% preassignment distribution parameters

InternationalCount = preassignment[['CrewID','InternationalCount']].groupby(['CrewID']).sum().reset_index()
LandingCount = preassignment[['CrewID','NoOfLandings']].groupby(['CrewID']).sum().reset_index()
LayoverCount = preassignment[['CrewID','IsLayOver']].groupby(['CrewID']).sum().reset_index()
FTCount = preassignment[['CrewID','FT']].groupby(['CrewID']).sum().reset_index()

pre_int =  dict(zip(C, [0]*len(C)))
pre_layover = dict(zip(C, [0]*len(C)))
pre_landing = dict(zip(C, [0]*len(C)))
pre_FT_hour = dict(zip(C, [0]*len(C)))
pre_DI = {}
pre_AC = {}
# no pattern here, already done above

for ind in InternationalCount.index:
    pre_int.update({str(InternationalCount['CrewID'][ind]):int(InternationalCount['InternationalCount'][ind])})

for ind in LandingCount.index:
    pre_landing.update({str(LandingCount['CrewID'][ind]):int(LandingCount['NoOfLandings'][ind])})
    
for ind in LayoverCount.index:
    pre_layover.update({str(LayoverCount['CrewID'][ind]):int(LayoverCount['IsLayOver'][ind])})
    
for ind in FTCount.index:
    pre_FT_hour.update({str(FTCount['CrewID'][ind]):int(FTCount['FT'][ind])/60})
   
for i in C:
    for di in DI:
        pre_DI.update({(i,di):0})
        
for i in C:
    if int(i) in set(preassignment['CrewID']):
        if 360 in preassignment[preassignment['CrewID'] == int(i)]['DiurnalStart'].values:
            pre_DI.update({(i,'360'):preassignment[preassignment['CrewID'] == int(i)]['DiurnalStart'].value_counts()[360]})
        elif 1081 in preassignment[preassignment['CrewID'] == int(i)]['DiurnalStart'].values:
            pre_DI.update({(i,'1081'):preassignment[preassignment['CrewID'] == int(i)]['DiurnalStart'].value_counts()[1081]})
        
for i in C:
    for t in Type:
        pre_AC.update({(i,t):0})  
    
for i in C:
    if int(i) in set(preassignment['CrewID']):
        if 2 in preassignment[preassignment['CrewID'] == int(i)]['AircraftID'].values:
            pre_AC.update({(i,'2'):preassignment[preassignment['CrewID'] == int(i)]['AircraftID'].value_counts()[2]})
        elif 4 in preassignment[preassignment['CrewID'] == int(i)]['AircraftID'].values:
            pre_AC.update({(i,'4'):preassignment[preassignment['CrewID'] == int(i)]['AircraftID'].value_counts()[4]})
        elif 5 in preassignment[preassignment['CrewID'] == int(i)]['AircraftID'].values:
            pre_AC.update({(i,'5'):preassignment[preassignment['CrewID'] == int(i)]['AircraftID'].value_counts()[5]})

#%% pre rolling window
# delta_pre_n in preassignments
pretrip_time = preassignment[['DutyID', 'DutyStart','DutyEnd','TripId']].drop_duplicates().dropna().reset_index()
def get_delta_pre_n(timewindowlist):
    delta_pre_n = {}
    for pp in pre:
        for n in timewindowlist:
            delta_pre_n.update({(pp,n):0})
    for ind in pretrip_time.index:
        for n in timewindowlist:
            start_date = datetime.strptime(pretrip_time['DutyStart'][ind], '%Y-%m-%dT%H:%M:%S').date()
            end_date = datetime.strptime(pretrip_time['DutyEnd'][ind], '%Y-%m-%dT%H:%M:%S').date()
            if (start_date in n) or (end_date in n):
                delta_pre_n.update({(str(pretrip_time['DutyID'][ind]),n):1})
    return delta_pre_n


# deal with preassignments in timewindow n
def pre_hour_within_window(timewindowlist):
    t_pre = {}
    for pp in pre:
        for n in timewindowlist:
            t_pre.update({(pp,n):0})
    for ind in preassignment.index:
        for n in timewindowlist:
            start_trip = datetime.strptime(preassignment['DutyStart'][ind], '%Y-%m-%dT%H:%M:%S')
            end_trip = datetime.strptime(preassignment['DutyEnd'][ind], '%Y-%m-%dT%H:%M:%S')
            start_window = date_converter.date_to_datetime(n[0])
            end_window = date_converter.date_to_datetime(n[len(n)-1])
            if (start_trip > start_window) and (end_trip < end_window):
                interval_1 = end_trip - start_trip
                hours_1 = interval_1.days*24+interval_1.seconds/3600
                t_pre.update({(str(preassignment['DutyID'][ind]),n):hours_1})
            elif (start_trip < start_window) and (end_trip > start_window) and (end_trip < end_window):
                interval_2 = end_trip - start_window
                hours_2 = interval_2.days*24+interval_2.seconds/3600
                t_pre.update({(str(preassignment['DutyID'][ind]),n):hours_2})
            elif (start_trip > start_window) and (start_trip < end_window) and (end_trip > end_window):
                interval_3 = end_window - start_trip
                hours_3 = interval_3.days*24+interval_3.seconds/3600
                t_pre.update({(str(preassignment['DutyID'][ind]),n):hours_3})
            elif (start_trip < start_window) and (end_trip > end_window):
                interval_4 = end_window - start_window
                hours_4 = interval_4.days*24+interval_4.seconds/3600
                t_pre.update({(str(preassignment['DutyID'][ind]),n):hours_4})
    return t_pre

n1 = T7_window[12]
pre_hour_within_window(T7_window)[('4135',n1)]

# which crew in which pre trip
h_jpre = {}
h_jpre = {(i,pp):0 for i in C for pp in pre}
# for i in C:
#     for pp in pre_trip:
#         if ((preassignment['CrewID'] == int(i)) & (preassignment['DutyID'] == int(pp))).any() == True:
#             h_jpre.update({(i,pp):1}) 
#         else:
#             h_jpre.update({(i,pp):0}) 
for u,v in h_jpre.items():
    if u in D:
        h_jpre[u]=1
        
h_jpre[('598', '381')]          

#%% Final input of Laminaar

T7 = get_rolling_time_window(date(2020,7,1),date(2020,7,31),7)
T14 = get_rolling_time_window(date(2020,7,1),date(2020,7,31),14)
T30 = get_rolling_time_window(date(2020,7,1),date(2020,7,31),30)
T365 = get_rolling_time_window(date(2020,7,1),date(2020,7,31),365)

delta_fn7 = get_delta_fn(T7)
delta_fn30 = get_delta_fn(T30)
delta_fn365 = get_delta_fn(T365)
delta_jn7 = get_delta_dn(T7)
delta_jn14 = get_delta_dn(T14)
delta_pre_n7 = get_delta_pre_n(T7)
delta_pre_n14 = get_delta_pre_n(T14)

d_7 = duty_hour_within_window(T7)
d_14 = duty_hour_within_window(T14)

t_pre_7 = pre_hour_within_window(T7)
t_pre_14 = pre_hour_within_window(T14)

M = 100000
FT7 = 34
DP7 = 60
DP14 = 100
FT30 = 110
FT365 = 980

# n1 = T7[10]

# for d in flight_duty:
#     print(d,d_7[(d,n1)])
    
Q_dict_wk = {}
for q in Q:
    Q_dict_wk.update({(q[0],q[1]):q[2]})
Q_dict_wk1 = dict(zip(list(Q_dict_wk.keys()),[()]*len(list(Q_dict_wk.keys()))))
for q in Q:
    Q_dict_wk1[(q[0],q[1])] = Q_dict_wk1[(q[0],q[1])]+(q[2],) 
    

Q_dict_pt = {}
for q in Q:
    Q_dict_pt.update({(q[0],q[2]):q[1]})
Q_dict_pt2 = dict(zip(list(Q_dict_pt.keys()),[()]*len(list(Q_dict_pt.keys()))))

for q in Q:
    Q_dict_pt2[(q[0],q[2])] = Q_dict_pt2[(q[0],q[2])]+(q[1],)
    

#%%  version 1 objective

model = ConcreteModel()
model.x_ij = Var(D, within = Binary)
model.x_if = Var(C, F, within = Binary)
model.x_id = Var(C, flight_duty, within = Binary)
model.s_ij = Var(D, within = Binary)
model.e_ij = Var(D, within = Binary)
model.BTPast_i=Var(C, within=NonNegativeReals)
model.LastFlown_ip=Var(C,Pt,within=NonNegativeReals)
model.NDF_ipt=Var(C,Pt,within=NonNegativeReals)
model.BT_i=Var(C,within=NonNegativeReals)
model.Int_i=Var(C,within=NonNegativeReals)
model.AP_id=Var(C,DI,within=NonNegativeReals)
model.AC_it=Var(C,Type,within=NonNegativeReals)
model.Land_i=Var(C,within=NonNegativeReals)
model.Lay_i=Var(C,within=NonNegativeReals) 
model.Slack_j=Var(P,within=Binary)
    
model.y_ijj_prime = Var(Q, within = Binary)
model.z_ij=Var(D,within=NonNegativeReals)
model.w_ijp=Var(D,Pt,within=NonNegativeReals)
model.v_ijjp=Var(Q,Pt,within=NonNegativeReals)

'''
J1 = w_BTPast*sum([(t_BTPast-model.BTPast_i[i])**2 for i in C])
J2 = w_LastFlown*sum([(t_LastFlown[p]-model.LastFlown_ip[i,p])**2 for i in C for p in Pt])
J3 = w_NDF*sum([(t_NDF[p]-model.NDF_ipt[i,p])**2 for i in C for p in Pt])
J4 = w_BT*sum([(t_BT-model.BT_i[i])**2 for i in C])
J5 = w_other*sum([(t_other-model.BT_i[i])**2 for i in C])
J6 = w_international*sum([(t_international-model.Int_i[i])**2 for i in C])
J7 = w_DI*sum([(t_DI[di]-model.AP_id[i,di])**2 for i in C for di in DI])
J8 = w_AC*sum([(t_AC[ac]-model.AC_it[i,ac])**2 for i in C for ac in Type])
J9 = w_Land*sum([(t_Land-model.Land_i[i])**2 for i in C])
J10 = w_Lay*sum([(t_Lay-model.Lay_i[i])**2 for i in C])
'''


def objective_rule(model):
    return sum([penalty_j[j]*model.Slack_j[j] for j in P])#+J1+J2+J3+J4+J5+J6+J7+J8+J9+J10
model.objective = Objective(rule=objective_rule, sense=minimize)

model.crew=ConstraintList() 
#%% Linear estimation of by FOTA
'''
model = ConcreteModel()
model.x_ij = Var(D, within = Binary)
model.x_if = Var(C, F, within = Binary)
model.x_id = Var(C, flight_duty, within = Binary)
model.s_ij = Var(D, within = Binary)
model.e_ij = Var(D, within = Binary)
model.BTPast_i=Var(C, within=NonNegativeReals)
model.LastFlown_ip=Var(C,Pt,within=NonNegativeReals)
model.NDF_ipt=Var(C,Pt,within=NonNegativeReals)
model.BT_i=Var(C,within=NonNegativeReals)
model.Int_i=Var(C,within=NonNegativeReals)
model.AP_id=Var(C,DI,within=NonNegativeReals)
model.AC_it=Var(C,Type,within=NonNegativeReals)
model.Land_i=Var(C,within=NonNegativeReals)
model.Lay_i=Var(C,within=NonNegativeReals) 
model.Slack_j=Var(P,within=Binary)
    
model.y_ijj_prime = Var(Q, within = Binary)
model.z_ij=Var(D,within=NonNegativeReals)
model.w_ijp=Var(D,Pt,within=NonNegativeReals)
model.v_ijjp=Var(Q,Pt,within=NonNegativeReals)

model.J1 = Var(within=NonNegativeReals, initialize=0)
model.J2 = Var(within=NonNegativeReals, initialize=0)
model.J3 = Var(within=NonNegativeReals, initialize=0)
model.J4 = Var(within=NonNegativeReals, initialize=0)
model.J5 = Var(within=NonNegativeReals, initialize=0)
model.J6 = Var(within=NonNegativeReals, initialize=0)
model.J7 = Var(within=NonNegativeReals, initialize=0)
model.J8 = Var(within=NonNegativeReals, initialize=0)
model.J9 = Var(within=NonNegativeReals, initialize=0)
model.J10 = Var(within=NonNegativeReals, initialize=0)


def objective_rule(model):
    return sum([penalty_j[j]*model.Slack_j[j] for j in P])+model.J1+model.J2+model.J3+model.J4+model.J5+model.J6+model.J7+model.J8+model.J9+model.J10
model.objective = Objective(rule=objective_rule, sense=minimize)


model.crew=ConstraintList() 

for j in range(2,12):
    tj_val = list(t_j.values())
    sum_J1 = sum(tj_val[0:j-1])
    model.crew.add(model.J1 >= w_BTPast*sum([(t_BTPast - (pre_FT_hour[i] + pv_BTPast[i]+sum_J1))**2-2*(model.BTPast_i[i] - 
    (pre_FT_hour[i] + pv_BTPast[i]+sum_J1))*(t_BTPast - (pre_FT_hour[i] + pv_BTPast[i]+sum_J1)) for i in C]))
    
    model.crew.add(model.J4 >= w_BT*sum([(t_BT - (pre_FT_hour[i] +sum_J1))**2-2*(model.BT_i[i] - 
    (pre_FT_hour[i] + sum_J1))*(t_BT - (pre_FT_hour[i] +sum_J1)) for i in C]))
    
    model.crew.add(model.J5 >= w_other*sum([(t_other - (pre_FT_hour[i] +sum_J1))**2-2*(model.BT_i[i] - 
    (pre_FT_hour[i] + sum_J1))*(t_other - (pre_FT_hour[i] +sum_J1)) for i in C]))


for lf_0 in range(20,42): # not sure of the values
    model.crew.add(model.J2 >= w_LastFlown*sum([(t_LastFlown[p] - lf_0)**2-2*(model.LastFlown_ip[i,p] - lf_0)*(t_LastFlown[p] - lf_0)  for i in C for p in Pt]))
    
for NDF_0 in range(1,12):
    model.crew.add(model.J3 >= w_NDF*sum([(t_NDF[p] - (pre_pattern_number[(i,p)]  + pv_pattern_crew[(i,p)] + NDF_0))**2-2*(model.NDF_ipt[i,p] - 
    (pre_pattern_number[(i,p)]  + pv_pattern_crew[(i,p)] + NDF_0))*(t_NDF[p] - (pre_pattern_number[(i,p)]  + pv_pattern_crew[(i,p)] + NDF_0)) for i in C for p in Pt]))

for I_0 in range(1,12):
    model.crew.add(model.J6 >= w_international*sum([(t_international - (pre_int[i] + pv_International[i]+I_0))**2-2*(model.Int_i[i] - 
    (pre_int[i] + pv_International[i]+I_0))*(t_international - (pre_int[i] + pv_International[i]+I_0)) for i in C]))
    
    
for di_0 in range(1,12):
    model.crew.add(model.J7 >= w_DI*sum([(t_DI[d] - (pre_DI[(i,d)] + pv_DI_crew[(i,d)] + di_0))**2-2*(model.AP_id[i,d] - 
    (pre_DI[(i,d)] + pv_DI_crew[(i,d)] + di_0))*(t_DI[d] - (pre_DI[(i,d)] + pv_DI_crew[(i,d)] + di_0)) for i in C for d in DI]))    
 
for ac_0 in range(1,12):
    model.crew.add(model.J8 >= w_AC*sum([(t_AC[ac] - (pre_AC[(i,ac)] + pv_AC_crew[(i,ac)] + ac_0))**2-2*(model.AC_it[i,ac] - 
    (pre_AC[(i,ac)] + pv_AC_crew[(i,ac)] + ac_0))*(t_AC[ac] - (pre_AC[(i,ac)] + pv_AC_crew[(i,ac)] + ac_0)) for i in C for ac in Type]))
       
for j in range(2,12):
    ldj_val = list(ld_j.values())
    sum_J9 = sum(ldj_val[0:j-1])
    model.crew.add(model.J9 >= w_Land*sum([(t_Land - (pre_landing[i] + pv_Landing[i]+sum_J9))**2-2*(model.Land_i[i] - 
    (pre_landing[i] + pv_Landing[i]+sum_J9))*(t_BTPast - (pre_landing[i] + pv_Landing[i]+sum_J9)) for i in C]))


for ly_0 in range(20,30):
    model.crew.add(model.J10 >= w_Lay*sum([(t_Lay - (pre_layover[i] + pv_Lay[i]+ly_0))**2-2*(model.Lay_i[i] - 
    (pre_layover[i] + pv_Lay[i]+ly_0))*(t_Lay - (pre_layover[i] + pv_Lay[i]+ly_0)) for i in C]))
    

'''
 


#%% version 2 objective linear

'''
model = ConcreteModel()
model.x_ij = Var(D, within = Binary)
model.x_if = Var(C, F, within = Binary)
model.x_id = Var(C, flight_duty, within = Binary)
model.s_ij = Var(D, within = Binary)
model.e_ij = Var(D, within = Binary)
model.BTPast_i=Var(C, within=NonNegativeReals)
model.LastFlown_ip=Var(C,Pt,within=NonNegativeReals)
model.NDF_ipt=Var(C,Pt,within=NonNegativeReals)
model.BT_i=Var(C,within=NonNegativeReals)
model.Int_i=Var(C,within=NonNegativeReals)
model.AP_id=Var(C,DI,within=NonNegativeReals)
model.AC_it=Var(C,Type,within=NonNegativeReals)
model.Land_i=Var(C,within=NonNegativeReals)
model.Lay_i=Var(C,within=NonNegativeReals) 
model.Slack_j=Var(P,within=Binary)
    
model.y_ijj_prime = Var(Q, within = Binary)
model.z_ij=Var(D,within=NonNegativeReals)
model.w_ijp=Var(D,Pt,within=NonNegativeReals)
model.v_ijjp=Var(Q,Pt,within=NonNegativeReals)

model.J1 = Var(within=NonNegativeReals, initialize=0)
model.J2 = Var(within=NonNegativeReals, initialize=0)
model.J3 = Var(within=NonNegativeReals, initialize=0)
model.J4 = Var(within=NonNegativeReals, initialize=0)
model.J5 = Var(within=NonNegativeReals, initialize=0)
model.J6 = Var(within=NonNegativeReals, initialize=0)
model.J7 = Var(within=NonNegativeReals, initialize=0)
model.J8 = Var(within=NonNegativeReals, initialize=0)
model.J9 = Var(within=NonNegativeReals, initialize=0)
model.J10 = Var(within=NonNegativeReals, initialize=0)

model.cost=Objective(expr=model.J1+model.J2+model.J3+model.J4+model.J5+model.J6+model.J7+model.J8+model.J9+
                      model.J10+sum([penalty_j[j]*model.Slack_j[j] for j in P]),sense=minimize)


model.crew = ConstraintList()
model.crew.add(model.J1 >= w_BTPast*sum([t_BTPast-model.BTPast_i[i] for i in C]))
model.crew.add(model.J1 >= -w_BTPast*sum([t_BTPast-model.BTPast_i[i] for i in C]))
model.crew.add(model.J2 >= w_LastFlown*sum([t_LastFlown[p]-model.LastFlown_ip[i,p] for i in C for p in Pt]))
model.crew.add(model.J2 >= -w_LastFlown*sum([t_LastFlown[p]-model.LastFlown_ip[i,p] for i in C for p in Pt]))
model.crew.add(model.J3 >= w_NDF*sum([t_NDF[p]-model.NDF_ipt[i,p] for i in C for p in Pt]))
model.crew.add(model.J3 >= -w_NDF*sum([t_NDF[p]-model.NDF_ipt[i,p] for i in C for p in Pt]))
model.crew.add(model.J4 >= w_BT*sum([t_BT-model.BT_i[i] for i in C]))
model.crew.add(model.J4 >= -w_BT*sum([t_BT-model.BT_i[i] for i in C]))
model.crew.add(model.J5 >= w_other*sum([t_other-model.BT_i[i] for i in C]))
model.crew.add(model.J5 >= -w_other*sum([t_other-model.BT_i[i] for i in C]))
model.crew.add(model.J6 >= w_international*sum([t_international-model.Int_i[i] for i in C]))
model.crew.add(model.J6 >= -w_international*sum([t_international-model.Int_i[i] for i in C]))
model.crew.add(model.J7 >= w_DI*sum([t_DI[di]-model.AP_id[i,di] for i in C for di in DI]))
model.crew.add(model.J7 >= -w_DI*sum([t_DI[di]-model.AP_id[i,di] for i in C for di in DI]))
model.crew.add(model.J8 >= w_AC*sum([t_AC[ac]-model.AC_it[i,ac] for i in C for ac in Type]))
model.crew.add(model.J8 >= -w_AC*sum([t_AC[ac]-model.AC_it[i,ac] for i in C for ac in Type]))
model.crew.add(model.J9 >= w_Land*sum([t_Land-model.Land_i[i] for i in C]))
model.crew.add(model.J9 >= -w_Land*sum([t_Land-model.Land_i[i] for i in C]))
model.crew.add(model.J10 >= w_Lay*sum([t_Lay-model.Lay_i[i] for i in C]))
model.crew.add(model.J10 >= -w_Lay*sum([t_Lay-model.Lay_i[i] for i in C]))                    

'''
#%%  version 3 objective, to min(max(7day_DP))
''' 
model = ConcreteModel()
model.x_ij = Var(D, within = Binary)
model.x_id = Var(C, flight_duty, within = Binary)
model.Slack_j=Var(P,within=Binary)
model.b = Var(within=NonNegativeReals, initialize=0)

def objective_rule(model):
    return model.b
model.objective = Objective(rule=objective_rule, sense=minimize)

model.crew=ConstraintList() 
for j in P:
    model.crew.add(sum([model.x_ij[(i,j)] for i in C if (i,j) in D]) == 1-model.Slack_j[j]) #1

for j in P:
    model.crew.add(model.Slack_j[j]==0) #new
    
for v in validcrewtrip:
    for d in flight_duty:
        if d in duty_in_trip[v[1]]:
            model.crew.add((model.x_id[v[0],d]-model.x_ij[v])==0) #2b
    
for i in C:
    for n7 in T7:
        if (sum([d_7[(d,n7)] for d in flight_duty])!=0):
            model.crew.add(model.b>=sum([d_7[(d,n7)]*model.x_id[i,d] for d in flight_duty])
                           +sum([t_pre_7[(pp,n7)]*h_jpre[(i,pp)] for pp in pre])) #5
            
solver = SolverFactory('gurobi')
#solver.options["mipgap"] = 0.2
solver.solve(model, tee=True)
print(solver) 
print(value(model.objective))
'''
#%% subproblem: how to guarantee the consecutively assignable trips

'''

model = ConcreteModel()
model.x_ij = Var(D, within = Binary)
model.x_if = Var(C, F, within = Binary)
model.x_id = Var(C, flight_duty, within = Binary)
model.s_ij = Var(D, within = Binary)
model.e_ij = Var(D, within = Binary)

model.Slack_j=Var(P,within=Binary)   
model.y_ijj_prime = Var(Q, within = Binary)





def objective_rule(model):
    return sum([penalty_j[j]*model.Slack_j[j] for j in P])
model.objective = Objective(rule=objective_rule, sense=minimize)

model.crew=ConstraintList() 

for j in P:
    model.crew.add(sum([model.x_ij[(i,j)] for i in C if (i,j) in D]) == 1-model.Slack_j[j]) #2

for j in P:
    model.crew.add(model.Slack_j[j]==0) #3

for v in validcrewtrip:
    for f in F:
        if f in trip_flight[v[1]]:
            model.crew.add(model.x_if[v[0],f] == model.x_ij[v]) #4
        
for v in validcrewtrip:
    for d in flight_duty:
        if d in duty_in_trip[v[1]]:
            model.crew.add(model.x_id[v[0],d] == model.x_ij[v]) #5
            
for p in pre_act:
    model.crew.add(model.x_ij[p]==1) #6

for i in C:
    for j1 in assignable_crew[i]:
        for j2 in assignable_crew[i]:
            if (j1 != j2) and (gamma_j1j2[(j1,j2)]==1):
                model.crew.add((model.x_ij[(i,j1)]+model.x_ij[(i,j2)])<=1) #7
                

for q in Q:
    model.crew.add(model.x_ij[(q[0],q[1])] +model.x_ij[(q[0],q[2])] >= 2*model.y_ijj_prime[q]) #8
    
                 
for d in D:
    if d in list(Q_dict_wk1.keys()):
        wk1_list = Q_dict_wk1[d]
        model.crew.add(model.e_ij[d]+sum([model.y_ijj_prime[(d[0],d[1],j_prime)] for j_prime in wk1_list]) == model.x_ij[d]) #10
    else:
        model.crew.add(model.e_ij[d] == model.x_ij[d]) #10b
                    
for d in D:
    if d in list(Q_dict_pt2.keys()):
        wk2_list = Q_dict_pt2[d]
        model.crew.add(model.s_ij[d]+sum([model.y_ijj_prime[d[0],j,d[1]] for j in wk2_list]) == model.x_ij[d]) #9
    else:
        model.crew.add(model.s_ij[d] == model.x_ij[d]) #9b

for i in C:
    model.crew.add(sum([model.s_ij[i,j] for j in all_act if (i,j) in D]) <=1) #11
    model.crew.add(sum([model.e_ij[i,j] for j in all_act if (i,j) in D]) <=1) #12


for d in D:
    model.crew.add(model.x_ij[d] >= model.s_ij[d]) #13
    model.crew.add(model.x_ij[d] >= model.e_ij[d]) #14
    
for i in C:
    model.crew.add(sum([model.s_ij[i,j] for j in all_act if (i,j) in D]) <= sum([model.x_ij[i,j] for j in all_act if (i,j) in D])) #
    model.crew.add(sum([model.e_ij[i,j] for j in all_act if (i,j) in D]) <= sum([model.x_ij[i,j] for j in all_act if (i,j) in D])) #
    model.crew.add(sum([model.s_ij[i,j] for j in all_act if (i,j) in D])*210 >= sum([model.x_ij[i,j] for j in all_act if (i,j) in D])) #
    model.crew.add(sum([model.e_ij[i,j] for j in all_act if (i,j) in D])*210 >= sum([model.x_ij[i,j] for j in all_act if (i,j) in D])) #

for q in pre_duty_intrip:
    model.crew.add(model.y_ijj_prime[q] == 1)     
    
solver = SolverFactory('gurobi')
#solver.options["mipgap"] = 0.4
#solver.options["MIPFocus"] = 1
solver.solve(model, tee=True)
print(solver)  
'''


#%%Constraints  4s    

for j in P:
    model.crew.add(sum([model.x_ij[(i,j)] for i in C if (i,j) in D]) == 1-model.Slack_j[j]) #1

for j in P:
    model.crew.add(model.Slack_j[j]==0) #new
    
#for i in C:
    #model.crew.add(sum([model.x_ij[(i,j)] for j in P if (i,j) in D])>=1) #new
    
for v in validcrewtrip:
    for f in F:
        if f in trip_flight[v[1]]:
            model.crew.add(model.x_if[v[0],f] == model.x_ij[v]) #4
        
for v in validcrewtrip:
    for d in flight_duty:
        if d in duty_in_trip[v[1]]:
            model.crew.add(model.x_id[v[0],d] == model.x_ij[v]) #5
            
            
for p in pre_act:
    model.crew.add(model.x_ij[p]==1) #3
'''    
for i in C:
    for j1 in P:
        for j2 in P:
            if ((i,j1) in D) and ((i,j2) in D) and (gamma_j1j2[(j1,j2)]==1):
                model.crew.add((model.x_ij[(i,j1)]+model.x_ij[(i,j2)])<=1) #4
'''
for i in C:
    for j1 in assignable_crew[i]:
        for j2 in assignable_crew[i]:
            if (j1 != j2) and (gamma_j1j2[(j1,j2)]==1):
                model.crew.add((model.x_ij[(i,j1)]+model.x_ij[(i,j2)])<=1) #4
                
#%%   rolling time window constraints  55s   
for i in C:
    for n7 in T7:
        if (sum([d_7[(d,n7)] for d in flight_duty])!=0):
            model.crew.add(sum([d_7[(d,n7)]*model.x_id[i,d] for d in flight_duty])
                           +sum([t_pre_7[(pp,n7)]*h_jpre[(i,pp)] for pp in pre]) <= DP7) #5
                                
for i in C:
    for n14 in T14:
        if (sum([d_14[(d,n14)] for d in flight_duty])!=0):
            model.crew.add(sum([d_14[(d,n14)]*model.x_id[i,d] for d in flight_duty])
                       +sum([t_pre_14[(pp,n14)]*h_jpre[(i,pp)] for pp in pre]) <= DP14) #6             

for i in C:
    for n7 in T7:
        #if (sum([tj_7[(j,n7)] for j in P if (i,j) in validcrewtrip])!=0):
        model.crew.add(sum([t_f[f]*delta_fn7[(f,n7)]*model.x_if[i,f] for f in F]) <= FT7) # 7
            
for i in C:
    for n30 in T30:
        model.crew.add(sum([t_f[f]*delta_fn30[(f,n30)]*model.x_if[i,f] for f in F])<= FT30) #8

for i in C:
    for n365 in T365:
        model.crew.add(sum([t_f[f]*delta_fn365[(f,n365)]*model.x_if[i,f] for f in F])<= FT365) #9

#%% weekly rests 27s
for d in D:
    model.crew.add(model.z_ij[d] <= 168) #10  

for d in D:
    model.crew.add(model.z_ij[d] >= g_j[d[1]] * model.s_ij[d])
'''  
for i in C:
    for q in Q1:
        if ((i,q[0]) in D) and ((i,q[1]) in D):
            model.crew.add(model.x_ij[i,q[0]]+model.x_ij[i,q[1]]>=2*model.y_ijj_prime[i,q]) #yes
'''

for q in Q:
    model.crew.add(model.z_ij[(q[0],q[2])]>=model.z_ij[(q[0],q[1])]*(1-r_ijj_prime[q]) + tau1_ijj_prime[q] + g_j[q[2]] - M*(1-model.y_ijj_prime[q]))   #11 


'''             
for i in C: 
    for j_prime in all_act:
        if (i,j_prime) in D:
            model.crew.add(sum([model.y_ijj_prime[(i,j,j_prime)] for j in all_act if (i,j,j_prime) in Q]) <= model.x_ij[(i,j_prime)])
            
            
for i in C: 
    for j in all_act:
        if (i,j) in D:
            model.crew.add(sum([model.y_ijj_prime[(i,j,j_prime)] for j_prime in all_act if (i,j,j_prime) in Q])<= model.x_ij[(i,j)])
''' 
 
for d in D:
    if d in list(Q_dict_wk1.keys()):
        wk1_list = Q_dict_wk1[d]
        model.crew.add(model.e_ij[d]+sum([model.y_ijj_prime[(d[0],d[1],j_prime)] for j_prime in wk1_list]) == model.x_ij[d]) #10
    else:
        model.crew.add(model.e_ij[d] == model.x_ij[d]) #10b
                    
for d in D:
    if d in list(Q_dict_pt2.keys()):
        wk2_list = Q_dict_pt2[d]
        model.crew.add(model.s_ij[d]+sum([model.y_ijj_prime[d[0],j,d[1]] for j in wk2_list]) == model.x_ij[d]) #9
    else:
        model.crew.add(model.s_ij[d] == model.x_ij[d]) #9b  

for i in C:
    model.crew.add(sum([model.s_ij[i,j] for j in all_act if (i,j) in D]) <=1)
    model.crew.add(sum([model.e_ij[i,j] for j in all_act if (i,j) in D]) <=1)


for d in D:
    model.crew.add(model.x_ij[d] >= model.s_ij[d])
    model.crew.add(model.x_ij[d] >= model.e_ij[d])
    
'''
for d in D:
        model.crew.add(sum([model.y_ijj_prime[(d[0],d[1],j_prime)] for j_prime in all_act if (d[0],d[1],j_prime) in Q]) <= model.x_ij[d]) #12
        
# time consuming            
for d in D:
    model.crew.add(sum([model.y_ijj_prime[(d[0],j,d[1])] for j in all_act if (d[0],j,d[1]) in Q]) <= model.x_ij[d]) #13
'''            
            
for q in Q:
    model.crew.add(model.x_ij[(q[0],q[1])] +model.x_ij[(q[0],q[2])] >= 2*model.y_ijj_prime[q]) #14

for i in C:
    model.crew.add(sum([model.s_ij[i,j] for j in all_act if (i,j) in D]) <= sum([model.x_ij[i,j] for j in all_act if (i,j) in D])) #
    model.crew.add(sum([model.e_ij[i,j] for j in all_act if (i,j) in D]) <= sum([model.x_ij[i,j] for j in all_act if (i,j) in D])) #
    model.crew.add(sum([model.s_ij[i,j] for j in all_act if (i,j) in D])*210 >= sum([model.x_ij[i,j] for j in all_act if (i,j) in D])) #
    model.crew.add(sum([model.e_ij[i,j] for j in all_act if (i,j) in D])*210 >= sum([model.x_ij[i,j] for j in all_act if (i,j) in D])) #


for q in pre_duty_intrip:
    model.crew.add(model.y_ijj_prime[q] == 1) 
    
#%% last flown trip pattern 10mins
    
for q in Q:
    for pt in Pt:
        model.crew.add(model.v_ijjp[q,pt]<=M*model.y_ijj_prime[q]) #15
        model.crew.add(model.v_ijjp[q,pt]<=model.w_ijp[q[0],q[1],pt]+M*(1-model.y_ijj_prime[q])) #16
        model.crew.add(model.v_ijjp[q,pt]>=model.w_ijp[q[0],q[1],pt]-M*(1-model.y_ijj_prime[q])) #17  

# time consuming
'''        
for i in C:
    for j_prime in all_act:
        for p in Pt:
            if (i,j_prime) in D:
                model.crew.add(model.w_ijp[i,j_prime,p] == sum([model.v_ijjp[i,j,j_prime,p]*(1-p_jp[(j,p)])+ tau_ijj_prime[i,j,j_prime]*model.y_ijj_prime[i,j,j_prime] 
                                     for j in P if (i,j,j_prime) in Q]))  #18     
'''

       
for p in Pt:
    for (i,j_prime) in D:
        if (i,j_prime) in list(Q_dict_pt2.keys()):
            Q_list = Q_dict_pt2[(i,j_prime)]
            model.crew.add(model.w_ijp[i,j_prime,p] == sum([model.v_ijjp[i,j,j_prime,p]*(1-p_jp[(j,p)])+ tau2_ijj_prime[i,j,j_prime]*model.y_ijj_prime[i,j,j_prime] for j in Q_list]))  #18
        else:
            model.crew.add(model.w_ijp[i,j_prime,p] == 0)

#%% distribution parameters
                
for i in C:
    model.crew.add(model.BTPast_i[i] == pre_FT_hour[i] + pv_BTPast[i] + sum([model.x_ij[i,j]*t_j[j] for j in P if (i,j) in validcrewtrip])) #yes
    model.crew.add(model.BT_i[i] == pre_FT_hour[i] + sum([model.x_ij[i,j]*t_j[j] for j in P if (i,j) in validcrewtrip])) #yes
    model.crew.add(model.Int_i[i] == pre_int[i] + pv_International[i] + sum([model.x_ij[i,j]*I_j[j] for j in P if (i,j) in validcrewtrip])) #yes
    model.crew.add(model.Land_i[i] == pre_landing[i] + pv_Landing[i] + sum([model.x_ij[i,j]*ld_j[j] for j in P if (i,j) in validcrewtrip]))#yes
    model.crew.add(model.Lay_i[i] == pre_layover[i] + pv_Lay[i] + sum([model.x_ij[i,j]*layover_j[j] for j in P if (i,j) in validcrewtrip]))#yes  
       
for i in C:
    for p in Pt:
        if p in Pt_all:
            model.crew.add(model.NDF_ipt[i,p] == pre_pattern_number[(i,p)] + pv_pattern_crew[(i,p)] + sum([model.x_ij[i,j]*p_jp[(j,p)] for j in P if (i,j) in validcrewtrip])) #yes
            model.crew.add(model.LastFlown_ip[i,p] == sum([model.w_ijp[i,j,p]*p_jp[(j,p)] for j in P if (i,j) in validcrewtrip]))
 
for i in C:
    for d in DI:
        model.crew.add(model.AP_id[i,d] == pre_DI[(i,d)] + pv_DI_crew[(i,d)] + sum([model.x_ij[i,j]*diur_jd[(j,d)] for j in P if (i,j) in validcrewtrip])) #yes
        
for i in C:
    for t in Type:
        model.crew.add(model.AC_it[i,t] == pre_AC[(i,t)] + pv_AC_crew[(i,t)] + sum([model.x_ij[i,j]*Ty_jt[(j,t)] for j in P if (i,j) in validcrewtrip])) #yes  

#%% solutions
solver = SolverFactory('gurobi')
#solver.options["mipgap"] = 0.4
solver.options["MIPFocus"] = 1
solver.solve(model, tee=True)
print(solver)           

#%%
'''
for v in D:
    if model.x_ij[v]() == 1 :
        #final_pair[v[0]] =  final_pair[v[0]]+(v[1],)
        print ('{}:{}'.format(v,model.x_ij[v]()))
 
for v in D:
    if model.x_ij[v]() == 0:
        print(v[0])
        
final_pair = dict.fromkeys(C, ())
for v in D:
    if model.x_ij[v]() ==1:
        final_pair[v[0]] =  final_pair[v[0]]+(v[1],)
        # print ('{}:{}'.format(v,model.x_ij[v]()))
final_assign = pd.DataFrame(list(final_pair.items()),columns = ['crew','trip'])
final_assign.to_csv('final_1.csv')

crew_trip_final = []
for v in D:
    if model.x_ij[v]() >= 0.5:
        start = triptime_all_act_alone[triptime_all_act_alone['actID'] == int(v[1])]['dt_start'].values[0]
        end = triptime_all_act_alone[triptime_all_act_alone['actID'] == int(v[1])]['dt_end'].values[0]
        tuple1 = (v[0],v[1],start,end)
        crew_trip_final.append(tuple1)
crew_trip_final_df = pd.DataFrame.from_records(crew_trip_final, columns =['crew', 'Trip','start', 'end']) 
crew_trip_final_df.to_csv('crew_trip_time4.csv')   


for j in P:
    if model.Slack_j[j]() == 1:
        print(j)
        
print(value(model.objective)) 

sum([penalty_j[j]*model.Slack_j[j]() for j in P])

for q in Q:
    if (model.x_ij[q[0],q[1]]() >= 0.5) and (model.x_ij[q[0],q[2]]() >= 0.5) and (model.y_ijj_prime[q]() >=0.5):
         print(q)
        
for d in D:
    if model.s_ij[d]() >= 0.5:
        print(d)
        
for d in D:
    if model.e_ij[d]() >= 0.5:
     print(d)     
  
last =[]
for i in C:
    for p in Pt:
        last.append(model.LastFlown_ip[i,p]())
'''
           