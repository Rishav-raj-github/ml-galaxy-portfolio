/**
 * ML Galaxy Portfolio - Interactive Canvas Simulators (2026 Edition)
 * Houses mathematical simulation engines and high-fidelity rendering loops for:
 * 1. Linear & Polynomial Regression Sandbox
 * 2. K-Means Clustering Sandbox
 * 3. MLP Neural Network Node Flow
 */

class RegressionSandbox {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.points = [];
        this.degree = 1;
        this.lambda = 0.0;
        this.resize();
        this.bindEvents();
        this.render();
    }

    resize() {
        const rect = this.canvas.parentNode.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = 300;
    }

    bindEvents() {
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Map to normalized coordinate space [-1, 1] for mathematical stability
            const normX = (x / this.canvas.width) * 2 - 1;
            const normY = -(y / this.canvas.height) * 2 + 1; // Invert y

            this.points.push({ x: normX, y: normY });
            this.render();
        });
    }

    clear() {
        this.points = [];
        this.render();
    }

    setDegree(d) {
        this.degree = d;
        this.render();
    }

    setLambda(l) {
        this.lambda = l;
        this.render();
    }

    solveRegression() {
        if (this.points.length < this.degree + 1) return null;

        const m = this.points.length;
        const d = this.degree;

        // 1. Build Vandermonde Matrix X (m x d+1)
        const X = [];
        const y = [];
        for (let i = 0; i < m; i++) {
            const row = [];
            for (let j = 0; j <= d; j++) {
                row.push(Math.pow(this.points[i].x, j));
            }
            X.push(row);
            y.push(this.points[i].y);
        }

        // 2. Linear algebra: Solve normal equations with L2 Regularization (Ridge)
        // theta = (X^T X + lambda I)^-1 X^T y
        const XT = this.transpose(X);
        const XTX = this.multiply(XT, X);
        
        // Add L2 penalty to diagonal (excluding intercept)
        for (let i = 1; i < XTX.length; i++) {
            XTX[i][i] += this.lambda;
        }

        const XTX_inv = this.invertMatrix(XTX);
        if (!XTX_inv) return null;

        const XTy = this.multiplyVector(XT, y);
        const theta = this.multiplyVector(XTX_inv, XTy);

        return theta;
    }

    // --- Basic Matrix Helpers ---
    transpose(A) {
        return A[0].map((_, c) => A.map(r => r[c]));
    }

    multiply(A, B) {
        const out = Array(A.length).fill(0).map(() => Array(B[0].length).fill(0));
        for (let r = 0; r < A.length; r++) {
            for (let c = 0; c < B[0].length; c++) {
                for (let k = 0; k < B.length; k++) {
                    out[r][c] += A[r][k] * B[k][c];
                }
            }
        }
        return out;
    }

    multiplyVector(A, v) {
        return A.map(row => row.reduce((sum, val, idx) => sum + val * v[idx], 0));
    }

    invertMatrix(M) {
        const n = M.length;
        // Augment matrix with Identity
        const A = M.map((row, i) => [...row, ...Array(n).fill(0).map((_, j) => i === j ? 1 : 0)]);
        
        for (let i = 0; i < n; i++) {
            let pivot = A[i][i];
            if (Math.abs(pivot) < 1e-10) {
                // Try pivoting
                let swapRow = -1;
                for (let r = i + 1; r < n; r++) {
                    if (Math.abs(A[r][i]) > 1e-10) { swapRow = r; break; }
                }
                if (swapRow === -1) return null; // Singular matrix
                [A[i], A[swapRow]] = [A[swapRow], A[i]];
                pivot = A[i][i];
            }

            for (let j = 0; j < 2 * n; j++) A[i][j] /= pivot;

            for (let r = 0; r < n; r++) {
                if (r === i) continue;
                const factor = A[r][i];
                for (let j = 0; j < 2 * n; j++) A[r][j] -= factor * A[i][j];
            }
        }
        return A.map(row => row.slice(n));
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw mathematical background grid
        this.ctx.strokeStyle = 'rgba(255,255,255,0.03)';
        this.ctx.lineWidth = 1;
        const gridSpacing = 40;
        for (let x = 0; x < this.canvas.width; x += gridSpacing) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        for (let y = 0; y < this.canvas.height; y += gridSpacing) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }

        // Draw Origin Axes
        this.ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        this.ctx.beginPath();
        this.ctx.moveTo(this.canvas.width / 2, 0);
        this.ctx.lineTo(this.canvas.width / 2, this.canvas.height);
        this.ctx.moveTo(0, this.canvas.height / 2);
        this.ctx.lineTo(this.canvas.width, this.canvas.height / 2);
        this.ctx.stroke();

        // Fit & Draw Regression Curve/Line
        const theta = this.solveRegression();
        if (theta) {
            this.ctx.strokeStyle = 'rgba(0, 242, 254, 0.8)';
            this.ctx.lineWidth = 3.5;
            this.ctx.shadowColor = 'rgba(0, 242, 254, 0.4)';
            this.ctx.shadowBlur = 10;
            this.ctx.beginPath();

            let first = true;
            for (let px = 0; px < this.canvas.width; px++) {
                const normX = (px / this.canvas.width) * 2 - 1;
                
                // Calculate y coordinate based on regression parameters
                let normY = 0;
                for (let j = 0; j <= this.degree; j++) {
                    normY += theta[j] * Math.pow(normX, j);
                }

                // Map back to canvas pixels
                const py = ((-normY + 1) / 2) * this.canvas.height;

                if (first) {
                    this.ctx.moveTo(px, py);
                    first = false;
                } else {
                    this.ctx.lineTo(px, py);
                }
            }
            this.ctx.stroke();
            this.ctx.shadowBlur = 0; // Reset shadow
        }

        // Draw Points with glowing cyan halos
        this.points.forEach(pt => {
            const px = ((pt.x + 1) / 2) * this.canvas.width;
            const py = ((-pt.y + 1) / 2) * this.canvas.height;

            this.ctx.fillStyle = '#00F2FE';
            this.ctx.shadowColor = 'rgba(0, 242, 254, 0.6)';
            this.ctx.shadowBlur = 8;
            this.ctx.beginPath();
            this.ctx.arc(px, py, 6, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.shadowBlur = 0;

        // Overlay guide if empty
        if (this.points.length === 0) {
            this.ctx.fillStyle = 'rgba(255,255,255,0.4)';
            this.ctx.font = '13px "Inter", sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText("Click inside this grid to distribute 2D coordinates", this.canvas.width / 2, this.canvas.height / 2 - 15);
            this.ctx.fillText("and see the fitted regression line in real-time.", this.canvas.width / 2, this.canvas.height / 2 + 5);
        }
    }
}

class KMeansSandbox {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.points = [];
        this.centroids = [];
        this.k = 3;
        this.colors = ['#FF0844', '#FFAA00', '#00F2FE', '#E040FB', '#0FF0B3'];
        this.resize();
        this.bindEvents();
        this.generateRandomPoints(60);
        this.initializeCentroids();
        this.render();
    }

    resize() {
        const rect = this.canvas.parentNode.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = 300;
    }

    bindEvents() {
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.points.push({ x, y, cluster: -1 });
            this.render();
        });
    }

    setK(val) {
        this.k = parseInt(val);
        this.initializeCentroids();
        this.resetAssignments();
        this.render();
    }

    generateRandomPoints(count = 60) {
        this.points = [];
        // Generate in 3 rough natural groups
        const centers = [
            { x: this.canvas.width * 0.25, y: this.canvas.height * 0.3 },
            { x: this.canvas.width * 0.75, y: this.canvas.height * 0.35 },
            { x: this.canvas.width * 0.5, y: this.canvas.height * 0.7 }
        ];

        for (let i = 0; i < count; i++) {
            const c = centers[i % 3];
            const angle = Math.random() * Math.PI * 2;
            const r = Math.random() * 55;
            this.points.push({
                x: Math.max(10, Math.min(this.canvas.width - 10, c.x + Math.cos(angle) * r)),
                y: Math.max(10, Math.min(this.canvas.height - 10, c.y + Math.sin(angle) * r)),
                cluster: -1
            });
        }
        this.initializeCentroids();
        this.render();
    }

    initializeCentroids() {
        this.centroids = [];
        if (this.points.length === 0) return;
        
        // K-Means++ style initialization
        const first = this.points[Math.floor(Math.random() * this.points.length)];
        this.centroids.push({ x: first.x, y: first.y });

        for (let i = 1; i < this.k; i++) {
            let maxDist = -1;
            let bestCandidate = null;

            this.points.forEach(pt => {
                let minCentDist = Infinity;
                this.centroids.forEach(c => {
                    const d = Math.pow(pt.x - c.x, 2) + Math.pow(pt.y - c.y, 2);
                    if (d < minCentDist) minCentDist = d;
                });

                if (minCentDist > maxDist) {
                    maxDist = minCentDist;
                    bestCandidate = pt;
                }
            });

            if (bestCandidate) {
                this.centroids.push({ x: bestCandidate.x, y: bestCandidate.y });
            }
        }
    }

    resetAssignments() {
        this.points.forEach(p => p.cluster = -1);
    }

    step() {
        if (this.points.length === 0 || this.centroids.length === 0) return;

        // Step 1: Assign points to nearest centroid
        let changed = false;
        this.points.forEach(p => {
            let minDist = Infinity;
            let nearestIdx = -1;

            this.centroids.forEach((c, idx) => {
                const dist = Math.pow(p.x - c.x, 2) + Math.pow(p.y - c.y, 2);
                if (dist < minDist) {
                    minDist = dist;
                    nearestIdx = idx;
                }
            });

            if (p.cluster !== nearestIdx) {
                p.cluster = nearestIdx;
                changed = true;
            }
        });

        // Step 2: Recalculate centroids
        const sums = Array(this.k).fill(0).map(() => ({ x: 0, y: 0, count: 0 }));
        this.points.forEach(p => {
            if (p.cluster !== -1) {
                sums[p.cluster].x += p.x;
                sums[p.cluster].y += p.y;
                sums[p.cluster].count++;
            }
        });

        sums.forEach((s, idx) => {
            if (s.count > 0) {
                this.centroids[idx].x = s.x / s.count;
                this.centroids[idx].y = s.y / s.count;
            }
        });

        this.render();
        return changed;
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw structural nodes grid
        this.ctx.strokeStyle = 'rgba(255,255,255,0.02)';
        this.ctx.lineWidth = 1;
        for (let x = 0; x < this.canvas.width; x += 40) {
            this.ctx.beginPath(); this.ctx.moveTo(x, 0); this.ctx.lineTo(x, this.canvas.height); this.ctx.stroke();
        }
        for (let y = 0; y < this.canvas.height; y += 40) {
            this.ctx.beginPath(); this.ctx.moveTo(0, y); this.ctx.lineTo(this.canvas.width, y); this.ctx.stroke();
        }

        // Draw Points
        this.points.forEach(pt => {
            const hasCluster = pt.cluster !== -1;
            this.ctx.fillStyle = hasCluster ? this.colors[pt.cluster] : 'rgba(255,255,255,0.5)';
            this.ctx.shadowColor = hasCluster ? this.colors[pt.cluster] : 'rgba(255,255,255,0.2)';
            this.ctx.shadowBlur = hasCluster ? 4 : 0;
            this.ctx.beginPath();
            this.ctx.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.shadowBlur = 0;

        // Draw Centroids
        this.centroids.forEach((c, idx) => {
            this.ctx.fillStyle = this.colors[idx];
            this.ctx.strokeStyle = '#FFFFFF';
            this.ctx.lineWidth = 2.5;
            this.ctx.shadowColor = this.colors[idx];
            this.ctx.shadowBlur = 12;
            
            // Draw as structured Diamond/Square
            this.ctx.beginPath();
            this.ctx.moveTo(c.x, c.y - 9);
            this.ctx.lineTo(c.x + 9, c.y);
            this.ctx.lineTo(c.x, c.y + 9);
            this.ctx.lineTo(c.x - 9, c.y);
            this.ctx.closePath();
            this.ctx.fill();
            this.ctx.stroke();
        });
        this.ctx.shadowBlur = 0;
    }
}

class NeuralNetworkSandbox {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.nodes = [];
        this.connections = [];
        this.pulses = [];
        this.layers = [4, 5, 4, 2];
        this.resize();
        this.buildNetwork();
        this.render();
        this.animate();
    }

    resize() {
        const rect = this.canvas.parentNode.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = 300;
    }

    buildNetwork() {
        this.nodes = [];
        this.connections = [];
        
        const layerCount = this.layers.length;
        const spacingX = this.canvas.width / (layerCount + 1);

        // Build Nodes
        for (let l = 0; l < layerCount; l++) {
            const nodeCount = this.layers[l];
            const spacingY = this.canvas.height / (nodeCount + 1);
            for (let n = 0; n < nodeCount; n++) {
                this.nodes.push({
                    id: `${l}_${n}`,
                    layer: l,
                    x: spacingX * (l + 1),
                    y: spacingY * (n + 1),
                    activation: 0.0,
                    targetActivation: 0.0
                });
            }
        }

        // Build Synaptic Connections
        for (let i = 0; i < this.nodes.length; i++) {
            const src = this.nodes[i];
            this.nodes.forEach(dest => {
                if (dest.layer === src.layer + 1) {
                    this.connections.push({
                        src: src,
                        dest: dest,
                        weight: Math.random() * 2 - 1
                    });
                }
            });
        }
    }

    propagate() {
        this.pulses = [];
        // Activate Input Layer nodes
        this.nodes.forEach(node => {
            if (node.layer === 0) {
                node.activation = 1.0;
                node.targetActivation = 1.0;
            } else {
                node.activation = 0.0;
                node.targetActivation = 0.0;
            }
        });

        // Trigger signal pulses starting from layer 0 to 1
        this.connections.forEach(conn => {
            if (conn.src.layer === 0) {
                this.pulses.push({
                    src: conn.src,
                    dest: conn.dest,
                    weight: conn.weight,
                    progress: 0.0,
                    speed: 0.03
                });
            }
        });
    }

    animate() {
        // Update pulses
        for (let i = this.pulses.length - 1; i >= 0; i--) {
            const p = this.pulses[i];
            p.progress += p.speed;

            if (p.progress >= 1.0) {
                // Pulse reached target; update target activation level
                p.dest.targetActivation = Math.min(1.0, p.dest.targetActivation + Math.abs(p.weight * p.src.activation) * 0.4);
                
                // Spawn next layer pulses if target layer completes arrival
                const currentLayer = p.dest.layer;
                if (currentLayer < this.layers.length - 1) {
                    this.connections.forEach(conn => {
                        if (conn.src.id === p.dest.id) {
                            // Check if next pulse is already spawned
                            const alreadyExists = this.pulses.some(oldP => oldP.src.id === conn.src.id && oldP.dest.id === conn.dest.id);
                            if (!alreadyExists) {
                                this.pulses.push({
                                    src: conn.src,
                                    dest: conn.dest,
                                    weight: conn.weight,
                                    progress: 0.0,
                                    speed: 0.03
                                });
                            }
                        }
                    });
                }
                
                // Remove completed pulse
                this.pulses.splice(i, 1);
            }
        }

        // Interpolate activation levels for smooth visuals
        this.nodes.forEach(node => {
            node.activation += (node.targetActivation - node.activation) * 0.1;
        });

        this.render();
        requestAnimationFrame(() => this.animate());
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // 1. Draw Static Synapses
        this.connections.forEach(conn => {
            const activeCoeff = (conn.src.activation + conn.dest.activation) / 2;
            this.ctx.strokeStyle = conn.weight > 0 
                ? `rgba(0, 242, 254, ${0.05 + activeCoeff * 0.3})`
                : `rgba(255, 8, 68, ${0.05 + activeCoeff * 0.3})`;
            this.ctx.lineWidth = 1 + Math.abs(conn.weight) * 2;
            this.ctx.beginPath();
            this.ctx.moveTo(conn.src.x, conn.src.y);
            this.ctx.lineTo(conn.dest.x, conn.dest.y);
            this.ctx.stroke();
        });

        // 2. Draw Active Pulsing signals
        this.pulses.forEach(p => {
            const px = p.src.x + (p.dest.x - p.src.x) * p.progress;
            const py = p.src.y + (p.dest.y - p.src.y) * p.progress;

            this.ctx.fillStyle = p.weight > 0 ? '#00F2FE' : '#FF0844';
            this.ctx.shadowColor = p.weight > 0 ? '#00F2FE' : '#FF0844';
            this.ctx.shadowBlur = 8;
            this.ctx.beginPath();
            this.ctx.arc(px, py, 4.5, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.shadowBlur = 0;

        // 3. Draw Nodes
        this.nodes.forEach(node => {
            // Gradient fill based on activation
            const intensity = Math.round(node.activation * 255);
            this.ctx.fillStyle = `rgb(${Math.max(15, intensity)}, ${Math.max(20, Math.round(intensity * 0.9 + 25))}, ${Math.max(35, Math.round(intensity * 0.6 + 50))})`;
            
            this.ctx.strokeStyle = node.activation > 0.1 
                ? 'rgba(0, 242, 254, 0.8)' 
                : 'rgba(255,255,255,0.2)';
            this.ctx.lineWidth = 2;

            if (node.activation > 0.1) {
                this.ctx.shadowColor = 'rgba(0, 242, 254, 0.5)';
                this.ctx.shadowBlur = 8 + node.activation * 8;
            }

            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, 9 + node.activation * 3, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();
            this.ctx.shadowBlur = 0; // Reset
        });
    }
}

// Global hookups
window.RegressionSandbox = RegressionSandbox;
window.KMeansSandbox = KMeansSandbox;
window.NeuralNetworkSandbox = NeuralNetworkSandbox;
