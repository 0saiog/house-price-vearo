# House price frontend

React and TypeScript client for the Vearo house-price service. It loads the supported cities from
`GET /locations`, validates the property form, submits it to `POST /predict`, and renders the INR
estimate on a dedicated result route.

From this directory:

```bash
cp .env.example .env
npm install
npm run dev
```

Use `npm run lint` and `npm run build` for the production checks. The complete setup, API contract,
model results, and architecture are documented in the [root README](../README.md).
