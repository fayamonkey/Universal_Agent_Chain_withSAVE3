# Universal Agent Chain Builder 🤖

Eine flexible Anwendung zum Erstellen und Ausführen von KI-Agenten-Ketten mit GPT-4. Entwickle deine eigenen Agenten-Ketten mit 2 bis 99 Agenten, die nacheinander arbeiten und aufeinander aufbauen.

## Features

- 🔗 Flexible Agenten-Ketten mit 2-99 Agenten
- 🎯 Individuelle System-Prompts für jeden Agenten
- 📊 Live-Fortschrittsanzeige
- 📝 Markdown-Export der Ergebnisse
- 🌐 Benutzerfreundliche Streamlit-Oberfläche
- 🚀 Einfache Installation mit setup.bat

## Installation

### Automatische Installation (Windows)

1. Klone das Repository
2. Führe `setup.bat` aus
3. Gib deinen OpenAI API Key ein, wenn du dazu aufgefordert wirst
4. Fertig! Die App erstellt automatisch Desktop-Verknüpfungen

### Manuelle Installation

1. Klone das Repository
2. Installiere die Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```
3. Erstelle eine `.env` Datei mit deinem OpenAI API Key:
   ```
   OPENAI_API_KEY=dein-api-key-hier
   ```
4. Starte die App:
   ```bash
   streamlit run agent_chain_app.py
   ```

## Verwendung

1. **Konfiguration**
   - Gib deinen OpenAI API Key ein
   - Wähle die Anzahl der gewünschten Agenten (2-99)

2. **Agenten-Setup**
   - Definiere System-Prompts für jeden Agenten
   - Jeder Agent erhält automatisch die Outputs der vorherigen Agenten

3. **Ausführung**
   - Gib deinen initialen Input ein
   - Klicke auf "Run Agent Chain"
   - Beobachte den Fortschritt in Echtzeit

4. **Ergebnisse**
   - Siehe die Outputs jedes Agenten in expandierbaren Sektionen
   - Lade die kompletten Ergebnisse als Markdown-Datei herunter

## Beispiel-Anwendungen

1. **Textanalyse-Kette**
   - Agent 1: Initiale Textanalyse
   - Agent 2: Sentiment-Analyse
   - Agent 3: Zusammenfassung
   - Agent 4: Empfehlungen

2. **Content-Creation-Kette**
   - Agent 1: Themenanalyse
   - Agent 2: Recherche
   - Agent 3: Outline-Erstellung
   - Agent 4: Content-Generierung
   - Agent 5: Optimierung

3. **Code-Review-Kette**
   - Agent 1: Code-Analyse
   - Agent 2: Bug-Suche
   - Agent 3: Optimierungsvorschläge
   - Agent 4: Dokumentationsverbesserungen
   - Agent 5: Test-Empfehlungen

## Technische Details

- Python 3.8+
- Streamlit für die Web-Oberfläche
- OpenAI GPT-4 API
- Modulares Design für einfache Erweiterbarkeit

## Lizenz

MIT License - Siehe LICENSE Datei

## Beitragen

Beiträge sind willkommen! Bitte erstelle einen Pull Request oder öffne ein Issue für Vorschläge und Verbesserungen.

made by Dirk Wonhoefer, AI Engineering, 2025

https://ai-engineering.ai