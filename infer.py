import cv2
import os
import gco
import argparse
import numpy as np
import pickle as pkl

from tqdm import tqdm, trange
from glob import glob
from scipy import signal
from opendr.camera import ProjectPoints
from sklearn.mixture import GaussianMixture

from util.visibility import VisibilityChecker
from util.labels import LABELS_REDUCED, LABEL_COMP, LABELS_MIXTURES, read_segmentation
from tex.texture import TextureData, Texture


def main(data_file, frame_dir, segm_dir, out_file, num_iter):
    # Step 1: Make unwraps

    data = pkl.load(open(data_file, 'rb'))

    segm_files = np.array(sorted(glob(os.path.join(segm_dir, '*.png')) + glob(os.path.join(segm_dir, '*.jpg'))))
    frame_files = np.array(sorted(glob(os.path.join(frame_dir, '*.png')) + glob(os.path.join(frame_dir, '*.jpg'))))

    vt = np.load('assets/basicModel_vt.npy')
    ft = np.load('assets/basicModel_ft.npy')
    f = np.load('assets/basicModel_f.npy')

    verts = data['vertices']

    camera_c = data['camera_c']
    camera_f = data['camera_f']
    width = data['width']
    height = data['height']

    camera = ProjectPoints(t=np.zeros(3), rt=np.array([-np.pi, 0., 0.]), c=camera_c, f=camera_f, k=np.zeros(5))

    visibility = VisibilityChecker(width, height, f)

    texture = TextureData(1000, f, vt, ft, visibility)

    isos, vises, iso_segms = [], [], []

    for i, (v, frame_file, segm_file) in enumerate(tqdm(zip(verts, frame_files, segm_files))):
        frame = cv2.imread(frame_file) / 255.
        segm = read_segmentation(segm_file) / 255.
        mask = np.float32(np.any(segm > 0, axis=-1))

        camera.set(v=v)

        vis, iso, iso_segm = texture.get_data(frame, camera, mask, segm)

        vises.append(vis)
        isos.append(iso)
        iso_segms.append(np.uint8(iso_segm * 255))

    # Step 2: Segm vote gmm

    iso_mask = cv2.imread('assets/tex_mask_1000.png', flags=cv2.IMREAD_GRAYSCALE) / 255.
    iso_mask = cv2.resize(iso_mask, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    voting = np.zeros((1000, 1000, len(LABELS_REDUCED)))

    gmms = {}
    gmm_pixels = {}

    for color_id in LABELS_REDUCED:
        gmms[color_id] = GaussianMixture(LABELS_MIXTURES[color_id])
        gmm_pixels[color_id] = []

    for frame, segm, vis in zip(isos, iso_segms, vises):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) / 255.
        tex_segm = read_segmentation(segm)
        tex_weights = 1 - vis
        tex_weights = np.sqrt(tex_weights)

        for i, color_id in enumerate(LABELS_REDUCED):
            if color_id != 'Unseen' and color_id != 'BG':
                where = np.all(tex_segm == LABELS_REDUCED[color_id], axis=2)
                voting[where, i] += tex_weights[where]
                gmm_pixels[color_id].extend(frame[where].tolist())

    for color_id in LABELS_REDUCED:
        if gmm_pixels[color_id]:
            print('GMM fit {}...'.format(color_id))
            gmms[color_id].fit(np.array(gmm_pixels[color_id]))

    for i, color_id in enumerate(LABELS_REDUCED):
        if color_id == 'Unseen' or color_id == 'BG':
            voting[:, i] = -10

    voting[iso_mask == 0] = 0
    voting[iso_mask == 0, 0] = 1

    unaries = np.ascontiguousarray((1 - voting / len(isos)) * 10)
    pairwise = np.ascontiguousarray(LABEL_COMP)

    seams = np.load('assets/basicModel_seams.npy')
    edge_idx = pkl.load(open('assets/basicModel_edge_idx_1000_.pkl', 'rb'))

    dr_v = signal.convolve2d(iso_mask, [[-1, 1]])[:, 1:]
    dr_h = signal.convolve2d(iso_mask, [[-1], [1]])[1:, :]

    where_v = iso_mask - dr_v
    where_h = iso_mask - dr_h

    idxs = np.arange(1000 ** 2).reshape(1000, 1000)
    v_edges_from = idxs[:-1, :][where_v[:-1, :] == 1].flatten()
    v_edges_to = idxs[1:, :][where_v[:-1, :] == 1].flatten()
    h_edges_from = idxs[:, :-1][where_h[:, :-1] == 1].flatten()
    h_edges_to = idxs[:, 1:][where_h[:, :-1] == 1].flatten()

    s_edges_from, s_edges_to = edges_seams(seams, 1000, edge_idx)

    edges_from = np.r_[v_edges_from, h_edges_from, s_edges_from]
    edges_to = np.r_[v_edges_to, h_edges_to, s_edges_to]
    edges_w = np.r_[np.ones_like(v_edges_from), np.ones_like(h_edges_from), np.ones_like(s_edges_from)]

    gc = gco.GCO()
    gc.create_general_graph(1000 ** 2, pairwise.shape[0], True)
    gc.set_data_cost(unaries.reshape(1000 ** 2, pairwise.shape[0]))

    gc.set_all_neighbors(edges_from, edges_to, edges_w)
    gc.set_smooth_cost(pairwise)
    gc.swap(-1)

    labels = gc.get_labels().reshape(1000, 1000)
    gc.destroy_graph()

    segm_colors = np.zeros((1000, 1000, 3), dtype=np.uint8)

    for i, color_id in enumerate(LABELS_REDUCED):
        segm_colors[labels == i] = LABELS_REDUCED[color_id]

    # Step 3: Stitch texture

    seams = np.load('assets/basicModel_seams.npy')
    mask = cv2.imread('assets/tex_mask_1000.png', flags=cv2.IMREAD_GRAYSCALE) / 255.

    segm_template = read_segmentation(segm_colors)

    num_labels = len(isos)
    texture = Texture(1000, seams, mask, segm_template, gmms)

    texture_agg = isos[0]
    visibility_agg = np.array(vises[0])

    tex, _ = texture.add_iso(texture_agg, visibility_agg, np.zeros_like(visibility_agg), inpaint=False)

    for i in trange(num_iter):
        rl = np.random.choice(num_labels)
        texture_agg, labels = texture.add_iso(isos[rl], vises[rl], rl, inpaint=i == (num_iter-1))

    print('saving {}...'.format(os.path.basename(out_file)))
    cv2.imwrite(out_file, np.uint8(255 * texture_agg))


def edges_seams(seams, tex_res, edge_idx):
    edges = np.zeros((0, 2), dtype=np.int32)

    for _, e0, _, e1 in seams:
        idx0 = np.array(edge_idx[e0][0]) * tex_res + np.array(edge_idx[e0][1])
        idx1 = np.array(edge_idx[e1][0]) * tex_res + np.array(edge_idx[e1][1])

        if len(idx0) and len(idx1):
            if idx0.shape[0] < idx1.shape[0]:
                idx0 = cv2.resize(idx0.reshape(-1, 1), (1, idx1.shape[0]), interpolation=cv2.INTER_NEAREST)
            elif idx0.shape[0] > idx1.shape[0]:
                idx1 = cv2.resize(idx1.reshape(-1, 1), (1, idx0.shape[0]), interpolation=cv2.INTER_NEAREST)

            edges_new = np.hstack((idx0.reshape(-1, 1), idx1.reshape(-1, 1)))
            edges = np.vstack((edges, edges_new))

    edges = np.sort(edges, axis=1)

    return edges[:, 0], edges[:, 1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_file',
        type=str,
        help="pkl data file")

    parser.add_argument(
        'frame_dir',
        type=str,
        help="Directory that contains frame files")

    parser.add_argument(
        'segm_dir',
        type=str,
        help="Directory that contains clothes segmentation files")

    parser.add_argument(
        'out_file',
        type=str,
        help="Texture output file (JPG or PNG)")

    parser.add_argument(
        '--iter', '-t', default=15, type=int,
        help="Texture optimization steps")

    args = parser.parse_args()
    main(args.data_file, args.frame_dir, args.segm_dir, args.out_file, args.iter)
