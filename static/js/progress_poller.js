class ProgressPoller {
    constructor(config) {
        this.ids = config.ids || [];
        this.url = config.url;
        this.intervalTime = config.interval || 2000;
        this.timer = null;
        this.completedIds = new Set();
        this.onUpdate = config.onUpdate || null; // Custom callback
    }

    start() {
        if (this.ids.length === 0) {
            console.log("No files to track");
            return;
        }
        console.log(`Starting poller for ${this.ids.length} files`);
        this.poll(); // Initial poll
        this.timer = setInterval(() => this.poll(), this.intervalTime);
    }

    stop() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
            console.log("Poller stopped");
        }
    }

    async poll() {
        const activeIds = this.ids.filter(id => !this.completedIds.has(id));

        if (activeIds.length === 0) {
            console.log("All files completed. Stopping poller.");
            this.stop();
            return;
        }

        try {
            const response = await fetch(`${this.url}?ids=${activeIds.join(',')}`);
            if (!response.ok) {
                console.error("Polling failed:", response.status);
                return;
            }

            const data = await response.json();
            
            // Use custom callback if provided, otherwise default UI update
            if (this.onUpdate) {
                this.onUpdate(data.files, this);
            } else {
                this.defaultUpdateUI(data.files);
            }
        } catch (error) {
            console.error("Polling error:", error);
        }
    }

    defaultUpdateUI(files) {
        files.forEach(f => {
            const prog = document.getElementById(`progress-${f.id}`);
            const stat = document.getElementById(`status-${f.id}`);
            const wrapper = document.getElementById(`file-wrapper-${f.id}`);

            // Update progress (native <progress> element)
            if (prog && prog.tagName === 'PROGRESS') {
                prog.value = f.progress;
            }

            // Update status text
            if (stat) {
                if (f.status === 'COMPLETED') {
                    stat.textContent = "Completed!";
                    stat.classList.add("status-success");
                    if (wrapper) wrapper.classList.add("done");
                    this.completedIds.add(f.id);
                } else if (f.status === 'FAILED') {
                    stat.textContent = "Failed";
                    stat.style.color = "#ef4444";
                    this.completedIds.add(f.id);
                } else if (f.status === 'PROCESSING') {
                    stat.textContent = `${f.progress}%`;
                } else {
                    stat.textContent = f.status;
                }
            }
        });
    }
}