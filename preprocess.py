import numpy as np
import scipy.io as sio
import glob
import six
from tqdm import tqdm

matdir = "./data/trainval/Annotations_Part/"
savedir = "./data/maskpkl/"
parts_list = ['hair', 'head', 'nose', 'mouth', 'neck', 'torso',
              'rebrow', 'lebrow', 'reye', 'leye', 'rear', 'lear',
              'ruarm', 'luarm', 'rlarm', 'llarm', 'ruleg', 'luleg', 'rlleg', 'llleg']


def matfilelist(directory):
    return glob.glob(directory + '*.mat')


def make_mask(matfile):
    d = sio.loadmat(matfile)
    dic = {}
    dic["filename"] = d["anno"][0, 0][0][0]
    dic["object_name"] = [d["anno"][0, 0][1][0, i][0][0] for i in range(d["anno"][0, 0][1].shape[1])]
    i = 0
    dic["person"] = {}
    for index, obj in enumerate(dic["object_name"]):
        if obj == "person":
            # dic["person_index"] = d["anno"][0, 0][1][0, index][1][0, 0]
            # dic["person"][str(i)] = {}
            if not d["anno"][0, 0][1][0, index][3].shape == (0, 0):
                dic["person"][str(i)] = {d["anno"][0, 0][1][0, index][3][0, j][0][0]:np.array(d["anno"][0, 0][1][0, index][3][0, j][1]) for j in range(d["anno"][0, 0][1][0, index][3].shape[1])}
                # for loop, if not list-loop
                # for j in range(d["anno"][0, 0][1][0, index][3].shape[1]):
                #     dic["person"][str(i)][d["anno"][0, 0][1][0, index][3][0, j][0][0]] = np.array(d["anno"][0, 0][1][0, index][3][0, j][1])
            else:
                print "something wrong, there is no parts", matfile
            dic["person"][str(i)]["mask"] = np.array(d["anno"][0, 0][1][0, index][2])
            i += 1
    return dic


# def make_mask(dic, insize=(300, 300)):
#     mask = np.zeros((len(parts_list), )+ insize)
#     n_person = len(dic["person"].keys())
#     if dic["person"]["0"]["mask"] == insize:
#         [parts_list.index(j) * d["person"]["0"][j] for i in dic["person"][j] for j in dic["person"]]


def save_batch(matfilelist, batchsize):
    data = {}
    data["imgfilelist"] = []
    data["orig_size"] = np.zeros((batchsize, 2))
    data["mask"] = np.zeros((batchsize, len(parts_list), 300, 300))
    for index, f in tqdm(enumerate(matfilelist)):
        d = makedic(f)
        data["imgfilelist"].append(d["filename"])
        data["orig_size"][index, :] = im_shape = d["person"]["0"]["mask"].shape
        
        data["mask"][index, :, :im_shape[0], :im_shape[1]] = make_mask(d)

        np.array([for p in in d["person"][k].keys()
                  for k in d["person"].keys()])

    savename = savedir + dic["filename"] + ".pkl"
    with open(savename, 'wb') as output:
        six.moves.cPickle.dump(data, output, -1)






if __name__ == "__main__":
    l = matfilelist(matdir)
    for f in tqdm(l):
        makedic(f)





'''
import numpy as np
import scipy.io as sio
import glob
import six


matdir = "./masks/"
savedir = "./masks_pkl/"


def makedic(matfile):
    d = sio.loadmat(matfile)
    dic = {}
    dic["filename"] = matfile[matfile.rfind("/") + 1:matfile.rfind(".")]
    dic["object_name"] = "person"
    if dic["object_name"] == "person":
        dic["object_mask"] = d["M"]
        savename = savedir + dic["filename"] + ".pkl"
        with open(savename, 'wb') as output:
            six.moves.cPickle.dump(dic, output, -1)


l = glob.glob(matdir + '*.mat')
for f in l:
    print f
    makedic(f)


'''

