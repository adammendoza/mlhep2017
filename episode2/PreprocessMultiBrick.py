#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue

parser = argparse.ArgumentParser(description='Preprocess multiple bricks in parallel.')
parser.add_argument('--input', required=True, dest="input", metavar="FILE", type=str,
                    help="input tracks file")
parser.add_argument('--output', required=True, dest="output", metavar="FILE", type=str,
                    help="output tracks file")
args = parser.parse_args()

def AddFields(tracks, field_names):
    n_tracks = tracks.shape[0]
    array = np.zeros(n_tracks)
    for name in field_names:
        tracks[name] = pd.Series(array, tracks.index)

def SetSimpleFields(tracks):
    axes = ['X', 'Y']
    for axis in axes:
        a_min = np.min(tracks[axis[0]])
        a_max = np.max(tracks[axis[0]])
        a_middle = (a_max + a_min)/2
        tracks[axis[0] + '_n'] = tracks[axis[0]] - a_middle
    tracks['R_n'] = np.sqrt(tracks['X_n']**2 + tracks['Y_n']**2)
    tracks['Phi_n'] = np.arctan2(tracks['Y_n'], tracks['X_n'])
    tracks['n_brick_tracks'] = tracks.shape[0]

def SetMacroFields(tracks, slices):
    n_tracks = tracks.shape[0]
    n_slices = len(slices)
    good_tracks = tracks[tracks['chi2'] < 1.5]

    for n in range(0, n_tracks):
        tr_pos = tracks.iloc[n][['X', 'Y', 'Z']].values
        near_tracks = good_tracks
        for k in range(n_slices-1, -1, -1):
            near_tracks = near_tracks[
                np.linalg.norm(near_tracks[["X", "Y", "Z"]].values - tr_pos, axis=1) < slices[k]]
            for prefix in ['Fwd', 'Bwd']:
                if prefix == 'Fwd':
                    sel_tracks = near_tracks[near_tracks["Z"] > tr_pos[2]]
                else:
                    sel_tracks = near_tracks[near_tracks["Z"] <= tr_pos[2]]
                pr = '{}{}_'.format(prefix, k+1)
                tracks[pr + 'n'].values[n] = sel_tracks.shape[0]
                if sel_tracks.shape[0] > 0:
                    tracks[pr + 'dTX'].values[n] = np.average(sel_tracks['TX']) - tracks.iloc[n]['TX']
                    tracks[pr + 'dTY'].values[n] = np.average(sel_tracks['TY']) - tracks.iloc[n]['TY']
                    tracks[pr + 'sTX'].values[n] = np.std(sel_tracks['TX'])
                    tracks[pr + 'sTY'].values[n] = np.std(sel_tracks['TY'])
                    tracks[pr + 'chi2'].values[n] = np.average(sel_tracks['chi2'])
                    tracks[pr + 's_chi2'].values[n] = np.std(sel_tracks['chi2'])


def UpdateBrickTracks(queue, tracks):
    macro_vars = ['n', 'dTX', 'dTY', 'sTX', 'sTY', 'chi2', 's_chi2']
    names = ['X_n', 'Y_n', 'R_n', 'Phi_n', 'n_brick_trk' ]
    slices = [2000., 5000., 10000.]
    n_slices = len(slices)

    for prefix in ['Fwd', 'Bwd']:
        for k in range(0, n_slices):
            for name in macro_vars:
                names.append('{}{}_{}'.format(prefix, k + 1, name))

    AddFields(tracks, names)
    SetSimpleFields(tracks)
    SetMacroFields(tracks, slices)
#    return tracks
    queue.put(tracks)


tracks = pd.read_csv(args.input, index_col=0, compression='gzip')

N = 20
K = 5

queue = Queue()
updated_tracks = None
for n in range(0, N):
	processes = []
	for k in range(0, K):
		brick = n * K + k + 1
		p = Process(target=UpdateBrickTracks, args=(queue, tracks[tracks['brick_number'] == brick].copy()))
		p.start()
		processes.append(p)

	for k in range(0, K):
		brick_tracks = queue.get()
		print "Processed brick {}. n_tracks = {}".format(np.unique(brick_tracks['brick_number'])[0], brick_tracks.shape[0])
		if updated_tracks is None:
			updated_tracks = brick_tracks
		else:
			updated_tracks = updated_tracks.append(brick_tracks, ignore_index=True)
	for p in processes:
		p.join()

updated_tracks = updated_tracks.sort_index()
updated_tracks.to_csv(args.output, header=True, compression='gzip')
print "Total n_tracks = {}".format(updated_tracks.shape[0])
