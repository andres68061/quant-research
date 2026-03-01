import { create } from "zustand";

interface ReplayState {
  frameIndex: number;
  playing: boolean;
  speed: number;
  totalFrames: number;
  setFrame: (i: number) => void;
  play: () => void;
  pause: () => void;
  setSpeed: (s: number) => void;
  setTotalFrames: (n: number) => void;
  advance: () => void;
}

export const useReplayStore = create<ReplayState>((set, get) => ({
  frameIndex: 0,
  playing: false,
  speed: 1,
  totalFrames: 0,
  setFrame: (i) => set({ frameIndex: i }),
  play: () => set({ playing: true }),
  pause: () => set({ playing: false }),
  setSpeed: (s) => set({ speed: s }),
  setTotalFrames: (n) => set({ totalFrames: n }),
  advance: () => {
    const { frameIndex, totalFrames } = get();
    set({ frameIndex: frameIndex >= totalFrames - 1 ? 0 : frameIndex + 1 });
  },
}));
