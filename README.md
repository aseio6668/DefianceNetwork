# DefianceNetwork - Decentralized Streaming Platform (Prototype)

**⚠️ This project is currently in an early, prototypical stage of development. The code represents an architectural skeleton and is not yet functional as a complete application. ⚠️**

DefianceNetwork is a decentralized internet streaming platform built in Rust. The goal is to combine P2P video/audio streaming, cryptocurrency integration, and a modular, cross-platform architecture.

## 🌟 Core Architectural Concepts

This repository contains the foundational code for the DefianceNetwork. The architecture is designed to be modular, with different functionalities separated into individual Rust crates.

### 🏗️ Project Structure & Implemented Crates

The project is organized as a Rust workspace. Here are the currently enabled crates and their status:

*   `defiance-core`: Defines the core data structures for the P2P network, streaming, and video processing. Contains a skeleton implementation of a `libp2p`-based network stack.
*   `defiance-audigy`: The most developed component. It defines the `.augy` metadata format for audio content and includes a parser and a simulated audio streaming engine.
*   `defiance-bridge`: Provides the architectural foundation for a multi-cryptocurrency bridge. It includes skeleton implementations for "Paradigm" and "Arceon" networks.
*   `defiance-cast`: A skeleton implementation for Chromecast support, with data structures for devices and sessions but no-op protocol implementation.
*   `defiance-discovery`: A functional peer discovery mechanism that can fetch a list of bootstrap nodes from a JSON file in a GitHub repository.

### ❌ Disabled Components

The following components, described in the original `README.md`, are currently **disabled** in the build due to being incomplete:

*   `defiance-ui`: The user interface.
*   `defiance-ml`: The machine learning components for network optimization.

## 🚀 Getting Started

### Prerequisites

*   Rust 1.70+ (latest stable recommended)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/aseio6668/DefianceNetwork.git
cd DefianceNetwork

# Build all enabled components
cargo build --release

# Run the integration tests to verify the build
cargo test -p integration-tests
```

**Note:** There is currently no runnable application. The project consists of library crates that are not yet integrated into a final executable.

## 📅 Proposed Roadmap

This roadmap outlines the next steps to move the project from a prototype to a functional application.

### Phase 1: Core Network & Streaming (Short-Term)

*   **Objective:** Create a functional P2P network that can stream data between two peers.
*   **Key Tasks:**
    1.  **Complete `defiance-core` network layer:**
        *   Implement the request-response protocol for direct peer communication.
        *   Flesh out the event loop to handle incoming `NetworkMessage`s.
        *   Wire the network layer to the `StreamingEngine`.
    2.  **Implement basic streaming in `StreamingEngine`:**
        *   Implement the logic for chunking data and sending it over the network.
        *   Implement the logic for receiving and reassembling chunks.
    3.  **Create a simple CLI application:**
        *   Create a new crate for a CLI application that can act as a broadcaster or a viewer.
        *   This will allow for testing the core functionality without a complex UI.

### Phase 2: Video & Audio Integration (Mid-Term)

*   **Objective:** Integrate actual video and audio processing into the streaming engine.
*   **Key Tasks:**
    1.  **Integrate a media library:**
        *   Choose and integrate a media library like `gstreamer` or `ffmpeg` for video and audio processing.
        *   This will involve uncommenting the dependencies in `Cargo.toml` and fixing any build issues.
    2.  **Implement `VideoEngine`:**
        *   Implement the video encoding and decoding logic.
        *   Connect the `VideoEngine` to the `StreamingEngine` to process and stream video data.
    3.  **Enhance `defiance-audigy`:**
        *   Move from simulated streaming to actual audio streaming over the P2P network.

### Phase 3: UI & Advanced Features (Long-Term)

*   **Objective:** Build a user interface and begin implementing the more advanced features.
*   **Key Tasks:**
    1.  **Re-enable `defiance-ui`:**
        *   Choose a UI framework (the original `egui` is a good option) and start building the user interface.
        *   Connect the UI to the core application logic.
    2.  **Implement `defiance-bridge`:**
        *   Flesh out the cryptocurrency bridge with actual blockchain integrations.
    3.  **Begin work on `defiance-ml`:**
        *   Start implementing the machine learning models for network optimization.

## 🤝 Contributing

This project is in its early stages, and contributions are welcome. Please focus on the short-term goals in the roadmap to help build a stable foundation.

### Contribution Guidelines

*   Follow Rust naming conventions and best practices.
*   Write tests for new features.
*   Update documentation for public APIs.
*   Submit PRs with clear descriptions and test coverage.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
