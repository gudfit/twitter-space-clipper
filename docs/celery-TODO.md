# Celery Backend Integration TODO

## Project Structure & Monorepo Consideration
- [x] Use monorepo: keep Streamlit app and Celery worker in the same project
- [x] Create `celery_worker/` directory at the project root
  - [x] Add `celery_app.py` (Celery app factory/config)
  - [x] Add `tasks.py` (Celery tasks stubs for transcription, quote generation, summarization)
  - [x] Add `__init__.py` (optional, for package completeness)

## Celery Setup
- [x] Add Celery app config in `celery_worker/celery_app.py`
- [x] Add task stubs in `celery_worker/tasks.py`
- [x] Add Celery to `requirements.txt` (pending)

## Broker & Backend
- [x] Default to Redis for broker and backend (configurable via env)
- [ ] Document broker/backend setup in README (pending)

## Refactor Processing Logic
- [ ] Move long-running functions (transcription, quote generation, summarization) into Celery tasks (pending)
- [ ] Ensure tasks save results and status to persistent storage (pending)

## Streamlit Integration
- [ ] Update Streamlit UI to submit jobs to Celery and track job/task IDs (pending)
- [ ] Poll for job status and display progress/results (pending)
- [ ] Add error handling and user feedback for job failures (pending)

## UI Improvements
- [x] Fix URL display and selection
  - [x] Show truncated URLs in media selector dropdown
  - [x] Place selected URL in editable input field
  - [x] Remove redundant URL displays
  - [x] Improve URL history loading and display
- [x] Improve media selection interface
  - [x] Add file size information
  - [x] Simplify media loading workflow
  - [x] Better organization of media selector and input fields
- [x] Fix content display after loading
  - [x] Show quotes, audio, and transcript tabs properly
  - [x] Maintain proper state across UI updates
  - [x] Fix regeneration of quotes for loaded media

## Monitoring & Admin
- [ ] Add Flower or similar Celery monitoring tool (optional, pending)

## Testing & Documentation
- [ ] Test end-to-end: job submission, processing, result retrieval, error handling (pending)
- [ ] Update documentation (README, PRD) with new architecture and usage instructions (pending)
- [ ] Document recent UI improvements and features

---

**Comments:**
- Project structure and Celery scaffolding are in place
- UI improvements have been completed, focusing on better URL handling and media selection
- Next steps: implement Celery tasks integration and update documentation
- Consider adding more UI improvements based on user feedback
