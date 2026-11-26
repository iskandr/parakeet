"""GrowCut image segmentation (one automaton step).

Original authors: Nathan Faggian, Stefan van der Walt, Aron Ahmadia, Olivier Grisel
https://github.com/stefanv/growcut_py
"""

import numpy as np

import parakeet as par


@par.jit
def growcut(image, state, window_radius):
  height, width = image.shape[0], image.shape[1]

  def attack_cell(i, j):
    winning_colony = state[i, j, 0]
    defense_strength = state[i, j, 1]
    for jj in range(max(0, j - window_radius), min(width, j + window_radius + 1)):
      for ii in range(max(0, i - window_radius), min(height, i + window_radius + 1)):
        if ii != i or jj != j:
          ssd = np.sum((image[i, j, :] - image[ii, jj, :]) ** 2)
          gval = 1.0 - np.sqrt(ssd) / np.sqrt(3.0)
          attack_strength = gval * state[ii, jj, 1]
          if attack_strength > defense_strength:
            defense_strength = attack_strength
            winning_colony = state[ii, jj, 0]
    return np.array([winning_colony, defense_strength])

  return np.array([[attack_cell(i, j) for j in range(width)] for i in range(height)])


def main(n: int = 48, window_radius: int = 5):
  image = np.zeros((n, n, 3))
  state = np.zeros((n, n, 2))
  state[0, 0, 0] = 1  # colony 1 has strength 1 at the corner
  state[0, 0, 1] = 1
  nxt = growcut(image, state, window_radius)
  assert nxt.shape == state.shape
  assert nxt[:, :, 0].sum() == (window_radius + 1) ** 2  # the colony spread over its window
  return nxt


if __name__ == "__main__":
  print(main()[:8, :8, 0])
