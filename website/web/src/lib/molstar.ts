import { resolveUniprot } from "./genes";

declare global {
  interface Window {
    molstar?: {
      Viewer: {
        create: (
          elementOrId: string | HTMLElement,
          options: Record<string, unknown>,
        ) => Promise<{
          plugin: {
            clear: () => Promise<void>;
            canvas3d?: {
              requestCameraReset: (opts?: Record<string, unknown>) => void;
              camera: {
                getSnapshot: () => Record<string, unknown>;
                setState: (state: Record<string, unknown>) => void;
              };
            };
            managers?: {
              camera?: {
                reset: (durationMs?: number) => void;
              };
            };
          };
          loadStructureFromUrl: (
            url: string,
            format: string,
            isBinary: boolean,
            options: Record<string, unknown>,
          ) => Promise<void>;
        }>;
      };
    };
  }
}

export type StructureViewer = {
  plugin: { clear: () => Promise<void> };
};

type FrameCanvas = {
  handleResize?: () => void;
  requestResize?: () => void;
  boundingSphereVisible?: { center?: ArrayLike<number>; radius?: number };
  camera: {
    getFocus: (
      target: ArrayLike<number>,
      radius: number,
    ) => {
      position: ArrayLike<number>;
      target: ArrayLike<number>;
      radius?: number;
    };
    setState: (state: Record<string, unknown>, durationMs?: number) => void;
  };
};

/** Pull the camera in without shrinking the clip sphere (that slices the chain). */
function frameStructure(canvas3d: FrameCanvas, zoom: number): boolean {
  canvas3d.handleResize?.();
  const sphere = canvas3d.boundingSphereVisible;
  const radius = sphere?.radius ?? 0;
  const center = sphere?.center;
  if (!(radius > 1) || !center || center.length < 3) return false;

  const focus = canvas3d.camera.getFocus(center, radius);
  const pos = focus.position;
  const tgt = focus.target;
  const dx = Number(pos[0]) - Number(tgt[0]);
  const dy = Number(pos[1]) - Number(tgt[1]);
  const dz = Number(pos[2]) - Number(tgt[2]);
  const dist = Math.hypot(dx, dy, dz) || 1;
  const next = Math.max(dist * zoom, radius + 8);
  canvas3d.camera.setState(
    {
      ...focus,
      radius,
      position: [
        Number(tgt[0]) + (dx / dist) * next,
        Number(tgt[1]) + (dy / dist) * next,
        Number(tgt[2]) + (dz / dist) * next,
      ],
    },
    0,
  );
  return true;
}

async function zoomIntoStructure(
  viewer: {
    plugin: {
      canvas3d?: object;
      managers?: { camera?: { reset: (durationMs?: number) => void } };
    };
  },
  zoom: number,
) {
  // Mol* is loaded from a script tag. The declared viewer type is narrower than
  // the camera helpers this zoom path uses.
  const canvas3d = viewer.plugin.canvas3d as FrameCanvas | undefined;
  if (!canvas3d) {
    viewer.plugin.managers?.camera?.reset?.(0);
    return;
  }
  // The scene commits after load. Frame once the bounding sphere exists.
  for (let i = 0; i < 12; i++) {
    if (frameStructure(canvas3d, zoom)) return;
    await new Promise((r) => setTimeout(r, 80));
  }
}

export async function loadAlphaFold(
  gene: string,
  mount: HTMLElement,
  options?: { zoom?: number },
): Promise<{
  viewer: StructureViewer;
  caption: string;
  source: "alphafold";
  accession: string;
}> {
  if (!window.molstar) throw new Error("Mol* failed to load");
  const accession = await resolveUniprot(gene);
  const af = await fetch(`https://alphafold.ebi.ac.uk/api/prediction/${accession}`);
  if (!af.ok) throw new Error(`AlphaFold DB ${af.status}`);
  const entry = (await af.json())[0];
  if (!entry) throw new Error("No AlphaFold model");

  mount.innerHTML = "";
  const viewer = await window.molstar.Viewer.create(mount, {
    layoutIsExpanded: false,
    layoutShowControls: false,
    viewportShowAnimation: false,
    viewportShowSelectionMode: false,
  });
  await viewer.loadStructureFromUrl(entry.bcifUrl, "mmcif", true, {
    representationParams: { theme: { globalName: "plddt-confidence" } },
    label: entry.modelEntityId,
  });
  // zoom < 1 moves closer than the default fit. 1 frames the whole chain.
  const zoom = options?.zoom ?? 0.5;
  await zoomIntoStructure(viewer, zoom);
  await new Promise((r) => setTimeout(r, 200));
  await zoomIntoStructure(viewer, zoom);

  const plddt = Number(entry.globalMetricValue || 0).toFixed(1);
  return {
    viewer,
    caption: `AlphaFold model · mean confidence ${plddt}`,
    source: "alphafold" as const,
    accession,
  };
}
