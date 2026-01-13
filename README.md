# Auto-Annotator

KI-gestützte Bildannotation für Aktivitätserkennung.

## Features

- 🤖 Automatische Objekterkennung mit TensorFlow.js (COCO-SSD)
- 🏷️ Aktivitätserkennung (Schweißen, Transport, Putzen, etc.)
- 📦 Batch-Labeling für Bildserien
- 💾 Export als ZIP mit YOLO-kompatiblen Annotationen
- 🔒 100% lokal - keine Daten werden an Server gesendet (DSGVO-konform)
- 📱 PWA-fähig - offline nutzbar

## Installation

```sh
# Repository klonen
git clone <YOUR_GIT_URL>

# In das Projektverzeichnis wechseln
cd <YOUR_PROJECT_NAME>

# Abhängigkeiten installieren
npm install

# Entwicklungsserver starten
npm run dev
```

## Technologien

- React + TypeScript
- Vite
- TensorFlow.js
- Tailwind CSS
- shadcn/ui
