# DefianceNetwork Quick Start Guide

Get up and running with DefianceNetwork in just a few minutes!

## 🚀 5-Minute Setup

### Step 1: Install Prerequisites

#### Windows
```powershell
# Install Rust
winget install Rustlang.Rustup

# Install GStreamer
winget install GStreamer.GStreamer

# Restart your terminal to reload PATH
```

#### macOS
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install GStreamer
brew install gstreamer gst-plugins-base gst-plugins-good
```

#### Linux (Ubuntu/Debian)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install GStreamer and dependencies
sudo apt update
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
                 libgstreamer-plugins-bad1.0-dev build-essential
```

### Step 2: Clone and Build

```bash
# Clone the repository
git clone https://github.com/aseio6668/DefianceNetwork.git
cd DefianceNetwork

# Build the project (this may take a few minutes)
cargo build --release

# Verify the build
cargo test
```

### Step 3: Run Your First Node

```bash
# Start a DefianceNetwork node
cargo run -p defiance-core

# In another terminal, start the Audigy player
cargo run -p defiance-audigy
```

🎉 **Congratulations!** You now have a DefianceNetwork node running!

## 🎵 Try Audigy (Audio Streaming)

### Basic Audio Playback
```bash
# Run the Audigy engine
cargo run -p defiance-audigy -- --help

# Play a sample audio file (if available)
cargo run -p defiance-audigy -- play samples/welcome.augy
```

### Create Your First .augy File
```bash
# Convert an audio file to .augy format
cargo run -p defiance-audigy -- convert input.mp3 output.augy \
    --title "My First Audio" \
    --author "Your Name" \
    --subject "Technology"
```

## 💰 Set Up Cryptocurrency Integration

### Paradigm (PAR) Setup
```bash
# Create a new Paradigm wallet
cargo run -p defiance-bridge -- paradigm create-wallet

# Check wallet balance
cargo run -p defiance-bridge -- paradigm balance

# Send a test transaction (testnet)
cargo run -p defiance-bridge -- paradigm send \
    --to PAR1234567890abcdef1234567890abcdef12345678 \
    --amount 10.0
```

### Arceon (ARC) Setup
```bash
# Create a new Arceon wallet
cargo run -p defiance-bridge -- arceon create-wallet

# Generate a new receiving address
cargo run -p defiance-bridge -- arceon new-address

# Check transaction history
cargo run -p defiance-bridge -- arceon history --limit 10
```

### Cross-Chain Bridge
```bash
# Initiate a PAR to ARC transfer
cargo run -p defiance-bridge -- bridge transfer \
    --from-network paradigm \
    --to-network arceon \
    --amount 50.0 \
    --from-address PAR... \
    --to-address ARC...

# Check bridge status
cargo run -p defiance-bridge -- bridge status <transfer-id>
```

## 🌐 Connect to the Network

### Join the P2P Network
Your node automatically connects to other DefianceNetwork peers using:

1. **Local Discovery**: Finds peers on your local network
2. **DHT Bootstrap**: Connects to the global peer network
3. **GitHub Fallback**: Uses seed nodes if other methods fail

### Check Network Status
```bash
# View connected peers
cargo run -p defiance-core -- peers list

# Check network health
cargo run -p defiance-core -- status

# View bandwidth usage
cargo run -p defiance-core -- stats
```

## 📱 Platform-Specific Features

### Desktop Features
- **System Tray**: Minimize to system tray
- **Notifications**: Desktop notifications for new content
- **File Associations**: Double-click .augy files to play

### Chromecast Support
```bash
# Discover available cast devices
cargo run -p defiance-cast -- discover

# Cast audio to a device
cargo run -p defiance-cast -- cast \
    --device "Living Room TV" \
    --content welcome.augy
```

## 🛠️ Development Mode

### Enable Debug Logging
```bash
# Set environment variable for detailed logs
export RUST_LOG=debug

# Run with debug output
cargo run -p defiance-core
```

### Run Tests
```bash
# Run all tests
cargo test

# Run specific component tests
cargo test -p defiance-core
cargo test -p defiance-audigy
cargo test -p defiance-bridge

# Run integration tests
cargo test -p integration-tests
```

### Development Tools
```bash
# Install useful development tools
cargo install cargo-watch cargo-audit cargo-deny

# Auto-rebuild on changes
cargo watch -x 'build --all'

# Security audit
cargo audit

# License and dependency checking
cargo deny check
```

## 🎨 User Interface

### Renaissance Theme
The DefianceNetwork UI uses an eco-friendly "Renaissance" theme:

- **Colors**: Earth tones, forest greens, warm golds
- **Typography**: Classic serif fonts for headers
- **Shapes**: Organic, rounded corners inspired by nature
- **Dark/Light**: Both modes available with nature themes

### Customize Your Experience
```bash
# Switch to dark mode
cargo run -p defiance-ui -- --theme dark

# Adjust audio quality
cargo run -p defiance-audigy -- --quality high

# Set bandwidth limits
cargo run -p defiance-core -- --max-upload 1MB --max-download 10MB
```

## 🔧 Configuration

### Config File Location
DefianceNetwork stores configuration in:
- **Windows**: `%APPDATA%\DefianceNetwork\config.toml`
- **macOS**: `~/Library/Application Support/DefianceNetwork/config.toml`
- **Linux**: `~/.config/DefianceNetwork/config.toml`

### Basic Configuration
```toml
[network]
port = 9080
max_peers = 50
enable_discovery = true

[audio]
quality = "high"
cache_size = "1GB"
offline_mode = true

[crypto]
enable_bridge = true
default_network = "paradigm"
auto_confirm = false

[ui]
theme = "renaissance-light"
minimize_to_tray = true
show_notifications = true
```

## 🆘 Troubleshooting

### Common Issues

#### "Failed to connect to network"
```bash
# Check firewall settings
# Ensure port 9080 is open for P2P connections

# Try different ports
cargo run -p defiance-core -- --port 9081

# Check network connectivity
cargo run -p defiance-core -- network-test
```

#### "Audio playback failed"
```bash
# Check GStreamer installation
gst-inspect-1.0 --version

# Test audio system
cargo run -p defiance-audigy -- audio-test

# Check audio devices
cargo run -p defiance-audigy -- list-devices
```

#### "Cryptocurrency transaction failed"
```bash
# Check network connection
cargo run -p defiance-bridge -- network-status

# Verify wallet balance
cargo run -p defiance-bridge -- balance --all

# Check transaction status
cargo run -p defiance-bridge -- transaction <tx-hash>
```

### Get Help
- **Documentation**: Check `docs/` directory for detailed guides
- **GitHub Issues**: https://github.com/aseio6668/DefianceNetwork/issues
- **Discussions**: https://github.com/aseio6668/DefianceNetwork/discussions
- **Discord**: [Join our community](https://discord.gg/defiancenetwork)

## 🎯 Next Steps

### Explore Features
1. **Create Content**: Upload your first video or audio content
2. **Join Communities**: Find interesting broadcasts and creators
3. **Earn Cryptocurrency**: Share bandwidth and earn PAR/ARC
4. **Cast to TV**: Stream content to your TV via Chromecast

### Advanced Usage
1. **Run a Relay Node**: Help other users connect to the network
2. **Develop Plugins**: Extend DefianceNetwork with custom features
3. **Contribute**: Join our open-source development community
4. **Bridge Operation**: Provide liquidity for cross-chain transfers

### Learning Resources
- **Architecture Guide**: `docs/ARCHITECTURE.md`
- **API Documentation**: Generated with `cargo doc --open`
- **Example Projects**: `examples/` directory
- **Video Tutorials**: [Coming Soon]

---

Welcome to the DefianceNetwork community! 🌟