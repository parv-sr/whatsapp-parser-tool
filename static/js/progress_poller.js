class ProgressPoller {
    constructor(config) {
        this.ids = config.ids || [];
        this.url = config.url;
        this.intervalTime = config.interval || 2000;
        this.timer = null;
        this.completedIds = new Set();
        this.onUpdate = config.onUpdate || null;
        this.requestInFlight = false;
        console.log(`[poller] initialized ids=${JSON.stringify(this.ids)} interval=${this.intervalTime}ms url=${this.url}`);
    }

    start() {
        if (this.ids.length === 0) {
            console.log("[poller] no files to track");
            return;
        }

        console.log(`[poller] starting for ${this.ids.length} files`);
        this.poll();
        this.timer = setInterval(() => this.poll(), this.intervalTime);
    }

    stop() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
            console.log("[poller] stopped");
        }
    }

    async poll() {
        if (this.requestInFlight) {
            console.log("[poller] skipping poll because previous request is still running");
            return;
        }

        const activeIds = this.ids.filter(id => !this.completedIds.has(id));
        if (activeIds.length === 0) {
            console.log("[poller] all files reached terminal status; stopping");
            this.stop();
            return;
        }

        this.requestInFlight = true;
        const pollUrl = `${this.url}?ids=${activeIds.join(',')}&_=${Date.now()}`;
        console.log(`[poller] request start activeIds=${activeIds.join(',')} url=${pollUrl}`);

        try {
            const response = await fetch(pollUrl, { cache: "no-store" });
            if (!response.ok) {
                console.error(`[poller] request failed status=${response.status}`);
                return;
            }

            const data = await response.json();
            console.log(`[poller] response ok file_count=${(data.files || []).length}`);

            if (this.onUpdate) {
                this.onUpdate(data.files || [], this);
            } else {
                this.defaultUpdateUI(data.files || []);
            }
        } catch (error) {
            console.error("[poller] polling error", error);
        } finally {
            this.requestInFlight = false;
            console.log("[poller] request complete");
        }
    }

    defaultUpdateUI(files) {
        files.forEach(f => {
            console.log(`[poller] default update file=${f.id} status=${f.status} progress=${f.progress}`);
            const meter = document.getElementById(`meter-${f.id}`);
            const stat = document.getElementById(`status-${f.id}`);
            const wrapper = document.getElementById(`file-wrapper-${f.id}`);

            if (meter) {
                const stage = Math.max(0, Math.min(5, f.stage || 0));
                const cells = meter.querySelectorAll('.stage-cell');
                cells.forEach((cell, idx) => {
                    cell.classList.toggle('on', idx < stage);
                });
            }

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
                } else if (f.status === 'CANCELLED') {
                    stat.textContent = "Cancelled";
                    stat.style.color = "#f59e0b";
                    this.completedIds.add(f.id);
                } else {
                    stat.textContent = f.stage_label || f.status;
                }
            }
        });
    }
}