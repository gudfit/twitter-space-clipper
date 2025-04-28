# Product Requirements Document (PRD): Celery Job Runner Integration

## Overview

This document outlines the requirements for integrating a Celery-based job runner into the project to handle long-running tasks such as transcription, quote generation, and summarization. The goal is to enable these processes to run asynchronously and persistently, independent of user sessions in the Streamlit frontend.

---

## Problem Statement

Currently, long-running tasks (e.g., audio transcription, quote extraction, summarization) are executed synchronously within the Streamlit app. If a user disconnects or navigates away, these jobs may be interrupted or lost. This leads to poor reliability and user experience for lengthy or resource-intensive operations.

---

## Objectives

- Decouple heavy processing tasks from the Streamlit UI.
- Ensure jobs continue running even if the user disconnects.
- Provide job status tracking and result retrieval for users.
- Support scaling and robust error handling for background jobs.

---

## Scope

### In Scope
- Integration of Celery as a distributed task queue.
- Refactoring of transcription, quote generation, and summarization to run as Celery tasks.
- Persistent storage of job status and results (e.g., in a database or on disk).
- Streamlit UI updates to submit jobs, poll for status, and display results.

### Out of Scope
- Real-time streaming of intermediate results (future enhancement).
- Multi-user authentication/authorization (unless required for job tracking).

---

## Functional Requirements

### 1. Job Submission
- Users can submit audio files or URLs for processing via the Streamlit UI.
- Upon submission, a Celery task is enqueued for:
  - Audio transcription
  - Quote extraction
  - Summarization
- The UI receives a unique job/task ID for tracking.

### 2. Job Processing
- Celery workers pick up jobs from the queue and execute them independently of the Streamlit process.
- Each job should:
  - Download/process the audio (if needed)
  - Transcribe audio to text
  - Generate quotes from the transcript
  - Summarize the content
- Intermediate and final results are saved to persistent storage (e.g., files, database).

### 3. Job Status & Result Retrieval
- Users can check the status of their jobs using the job/task ID.
- The UI polls for job status (e.g., pending, started, finished, failed).
- When complete, users can view/download the results (transcript, quotes, summary).
- Error messages are displayed if a job fails.

### 4. Scalability & Reliability
- The system must support running multiple jobs concurrently.
- Jobs should be retryable on failure (configurable retry policy).
- Workers can be scaled horizontally as needed.

### 5. Security & Resource Management
- Jobs should be isolated per user/session (if multi-user support is needed).
- Resource limits (e.g., max job duration, file size) should be configurable.

---

## Non-Functional Requirements

- **Performance:** Jobs should not block the Streamlit UI. Users should be able to submit new jobs while others are processing.
- **Persistence:** All job results and statuses must be stored durably.
- **Monitoring:** Admins should be able to monitor job queue health and worker status (e.g., via Flower or similar tools).
- **Extensibility:** The system should allow for future addition of new job types (e.g., translation, audio enhancement).

---

## Technical Considerations

- **Celery Broker:** Use Redis or RabbitMQ as the message broker.
- **Result Backend:** Use Redis, a database, or the filesystem for storing results.
- **Task Serialization:** Use JSON for task arguments/results.
- **Deployment:** Celery workers run as separate processes/containers from the Streamlit app.
- **Error Handling:** Implement robust error and exception handling in all tasks.

---

## User Flow

1. User submits a media URL or file in Streamlit.
2. Streamlit enqueues a Celery job and returns a job ID.
3. User sees job status (pending/processing/complete/failed) in the UI.
4. When complete, user can view/download transcript, quotes, and summary.

---

## Open Questions

- Should job results be deleted after a certain period?
- Is user authentication required for job/result access?
- What is the maximum supported file size/duration?
- Should users be notified (e.g., email) when jobs complete?

---

## Success Criteria

- Users can submit jobs and retrieve results even after disconnecting.
- Jobs are processed reliably and efficiently in the background.
- The system is scalable and maintainable for future enhancements. 