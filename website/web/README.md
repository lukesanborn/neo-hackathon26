# Blood JEPA web (React + TypeScript + Vite)

## Local

The dev server proxies `/api` to the deployed Modal endpoint.

```bash
cd website/web && npm install && npm run dev
```

Open http://127.0.0.1:5173

Sample uploads (real Replogle K562 test-split cells, plus one unperturbed control) live in `test_set_examples/` at the repo root. `manifest.json` records the true knockdown for each file.

## Production build (Modal)

```bash
cd website/web && npm run build
modal deploy modal_blood_gpt.py
```

`modal_blood_gpt.py` serves `web/dist` next to the GPU `/api/*` routes.
