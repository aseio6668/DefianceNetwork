# DefianceNetwork Project Roadmap

This document outlines the development roadmap for DefianceNetwork, detailing the planned features and the order in which they will be implemented.

## Phase 1: Core Network & Streaming (Short-Term)

**Objective:** Create a functional P2P network that can stream data between two peers. This phase is about building a solid, testable foundation.

### 1.1: Complete `defiance-core` Network Layer

*   **Task:** Implement the `libp2p` request-response protocol for direct peer communication.
*   **Task:** Flesh out the `P2PNetwork` event loop to handle all defined `NetworkMessage`s.
*   **Task:** Implement a robust peer management system, including connection/disconnection logic and peer scoring.
*   **Task:** Wire the `P2PNetwork` layer to the `StreamingEngine` so they can communicate.

### 1.2: Implement Basic Streaming in `StreamingEngine`

*   **Task:** Implement the logic for chunking a file or data stream into `ContentChunk`s.
*   **Task:** Implement the logic for sending chunks to viewers via the `P2PNetwork`.
*   **Task:** Implement the logic for receiving and reassembling chunks on the viewer's side.
*   **Task:** Create a basic buffering mechanism to handle network jitter.

### 1.3: Create a Simple CLI Application

*   **Task:** Create a new crate (`defiance-cli`) for a command-line application.
*   **Task:** Implement a "broadcast" mode in the CLI that can stream a local file.
*   **Task:** Implement a "view" mode in the CLI that can connect to a broadcast and save the streamed file locally.
*   **Task:** This CLI will serve as the primary tool for testing and debugging the core functionality.

## Phase 2: Video & Audio Integration (Mid-Term)

**Objective:** Integrate actual video and audio processing into the streaming engine to enable live streaming.

### 2.1: Integrate a Media Library

*   **Task:** Choose and integrate a media library (`gstreamer` or `ffmpeg` are the primary candidates).
*   **Task:** Resolve any build issues and ensure the media library can be compiled across all target platforms (Windows, Linux, macOS).
*   **Task:** Create a simple abstraction layer over the media library to handle common tasks like encoding, decoding, and transcoding.

### 2.2: Implement `VideoEngine`

*   **Task:** Implement the video encoding pipeline to take raw video frames and encode them into a streamable format (e.g., H.264).
*   **Task:** Implement the video decoding pipeline to take received video chunks and decode them into playable frames.
*   **Task:** Connect the `VideoEngine` to the `StreamingEngine` to process and stream video data in real-time.
*   **Task:** Implement adaptive bitrate streaming by creating multiple quality levels of the video stream.

### 2.3: Enhance `defiance-audigy`

*   **Task:** Move from the current simulated streaming model to actual audio streaming over the P2P network.
*   **Task:** Integrate the chosen media library for audio encoding and decoding.
*   **Task:** Implement a simple audio player in the CLI application to test audio streaming.

## Phase 3: UI & Advanced Features (Long-Term)

**Objective:** Build a user interface and begin implementing the more advanced features outlined in the original vision.

### 3.1: Re-enable `defiance-ui`

*   **Task:** Choose a UI framework (`egui` is the current placeholder, but others can be considered).
*   **Task:** Design and implement the basic UI layout, including a content browser, video player, and broadcast controls.
*   **Task:** Connect the UI to the core application logic, allowing users to browse and watch streams.

### 3.2: Implement `defiance-bridge`

*   **Task:** Flesh out the cryptocurrency bridge with actual blockchain integrations. This will likely involve using libraries for interacting with Bitcoin, Ethereum, and other target networks.
*   **Task:** Implement a secure wallet management system.
*   **Task:** Implement the logic for atomic swaps or other cross-chain transfer mechanisms.

### 3.3: Begin work on `defiance-ml`

*   **Task:** Research and design the machine learning models for network optimization (e.g., peer selection, latency prediction).
*   **Task:** Collect data from the P2P network to train the models.
*   **Task:** Integrate the trained models into the `defiance-core` network layer to improve streaming performance.

## Ongoing Tasks

*   **Testing:** Each new feature should be accompanied by a comprehensive suite of unit and integration tests.
*   **Documentation:** All public APIs should be well-documented.
*   **CI/CD:** The CI/CD pipeline should be maintained and expanded to include more checks (e.g., code formatting, clippy).
