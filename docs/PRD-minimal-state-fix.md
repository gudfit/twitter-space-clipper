# Minimal State Fix PRD

## Overview
This document outlines the minimal requirements for fixing state management issues introduced by the Celery worker multi-process refactor, with the goal of restoring end-to-end functionality.

## Problem Statement
The Celery worker multi-process refactor has broken state management, causing:
- Inconsistent state updates between processes
- Lost state transitions during task handoffs
- UI not reflecting actual process state
- Error states not being properly propagated

## Goals
1. Restore basic end-to-end functionality
2. Ensure reliable state updates from Celery tasks
3. Fix basic UI state display
4. Handle basic error cases

## Non-Goals
1. Implement full state machine architecture
2. Add advanced polling optimizations
3. Add comprehensive error recovery
4. Implement advanced UI features

## Technical Requirements

### State Storage
- Use Redis as central state store
- Simple key-value state schema
- Basic atomic updates

#### Minimal State Schema
```json
{
    "status": "DOWNLOADING",  // Current status
    "message": "Downloading media file...",  // User-friendly message
    "error": null,  // Error message if any
    "updated_at": "2024-05-01T21:47:30.299Z",  // Last update timestamp
    "metadata": {
        "url": "...",
        "space_id": "...",
        "task_id": "..."
    }
}
```

### Basic States
```python
STATES = {
    'INIT': 'Starting process...',
    'DOWNLOADING': 'Downloading media...',
    'TRANSCRIBING': 'Transcribing audio...',
    'GENERATING_QUOTES': 'Generating quotes...',
    'GENERATING_SUMMARY': 'Creating summary...',
    'ERROR': 'Error occurred',
    'COMPLETE': 'Process complete'
}
```

### Celery Integration
- Add state updates at task boundaries
- Ensure state updates are atomic
- Pass state between tasks
- Handle basic task failures

### UI Updates
- Simple polling mechanism (1-second interval)
- Basic error display
- Status message updates
- Loading indicators

### Error Handling
- Basic error state propagation
- Simple error messages
- Task failure handling
- State cleanup on errors

## Implementation Plan

### Phase 1: Core State Management
1. Implement Redis state storage
2. Add basic state update functions
3. Add state cleanup utilities
4. Test state persistence

### Phase 2: Celery Integration
1. Add state updates to task entry/exit points
2. Implement basic error handling
3. Add state passing between tasks
4. Test task chain state flow

### Phase 3: UI Integration
1. Implement basic polling
2. Add status display updates
3. Show error messages
4. Test UI responsiveness

## Testing Requirements
- Test basic state transitions
- Verify error handling
- Check UI updates
- Test task chain completion

## Success Criteria
1. Complete process runs successfully
2. UI shows correct status
3. Errors are displayed
4. State persists between restarts

## Timeline
- Phase 1: 1 day
- Phase 2: 1-2 days
- Phase 3: 1 day
- Testing: 1 day

Total: 4-5 days

## Associated TODOs

### Phase 1: Core State Management
- [ ] Implement Redis state storage functions
  - [ ] Add set_state function
  - [ ] Add get_state function
  - [ ] Add clear_state function
- [ ] Add state validation
  - [ ] Validate state values
  - [ ] Check timestamps
  - [ ] Verify metadata
- [ ] Create state cleanup utilities
  - [ ] Remove stale states
  - [ ] Clean up error states
  - [ ] Archive completed states

### Phase 2: Celery Integration
- [ ] Update task entry points
  - [ ] Add state updates
  - [ ] Pass state data
  - [ ] Handle initialization
- [ ] Update task exit points
  - [ ] Update completion state
  - [ ] Handle errors
  - [ ] Clean up resources
- [ ] Implement error handling
  - [ ] Catch task exceptions
  - [ ] Update error states
  - [ ] Clean up on failure

### Phase 3: UI Integration
- [ ] Implement status polling
  - [ ] Add polling function
  - [ ] Handle timeouts
  - [ ] Update UI elements
- [ ] Add error display
  - [ ] Show error messages
  - [ ] Add retry option
  - [ ] Clear errors
- [ ] Test end-to-end flow
  - [ ] Verify state updates
  - [ ] Check error handling
  - [ ] Test UI responsiveness

## Future Considerations
1. Migration path to full state machine
2. Performance optimization opportunities
3. Enhanced error recovery options
4. UI improvement possibilities

---

> 📝 **Note:** This minimal fix serves as a foundation for the enhanced state management system outlined in PRD-enhanced-state-management.md 