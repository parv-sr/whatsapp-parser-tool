class ProgressPoller{
    constructor(config){
        this.ids = config.ids || [];
        this.url = config.url;
        this.intervalTime = config.interval || 2000;
        this.timer = null;
        this.completedIds = new Set();
    }

    start() {
        if (this.ids.length == 0) return;
        this.poll();
        this.timer = setInterval(() => this.poll(), this.intervalTime);
    }

    stop() {
        if (this.timer) clearInterval(this.timer);
    }

    async poll() {
        const activeIds = this.ids.filter(id => !this.completedIds.has(id));

        if (activeIds.length == 0){
            console.log("All filer processed. Stopping poller.")
            this.stop()
            return;
        }

        try{
            const response = await fetch(`${this.url}?ids=${activeIds.join(',')}`);
            if (!response.ok) throw new Error("Network response not ok");

            const data = await response.json();
            this.updateUI(data.files);
        }catch (error) {
            console.error("Polling error: ", error)
        }
    }

    updateUI(files) {
        files.forEach(file => {
            // Update Progress Bar
            const bar = document.getElementById(`progress-${file.id}`);
            if (bar) {
                bar.style.width = `${file.progress}%`;
                bar.setAttribute('aria-valuenow', file.progress);
                
                // UX: Change color on completion/failure
                if (file.status === 'COMPLETED') {
                    bar.classList.remove('progress-bar-striped', 'progress-bar-animated', 'bg-primary');
                    bar.classList.add('bg-success');
                    this.completedIds.add(file.id);
                } else if (file.status === 'FAILED') {
                    bar.classList.remove('progress-bar-striped', 'progress-bar-animated');
                    bar.classList.add('bg-danger');
                    this.completedIds.add(file.id);
                }
            }

            // Update Status Badge/Text
            const badge = document.getElementById(`status-${file.id}`);
            if (badge) {
                badge.textContent = file.status;
                // Reset classes and apply new ones
                badge.className = 'badge status-badge'; 
                if (file.status === 'COMPLETED') badge.classList.add('bg-success');
                else if (file.status === 'FAILED') badge.classList.add('bg-danger');
                else if (file.status === 'PROCESSING') badge.classList.add('bg-primary');
                else badge.classList.add('bg-secondary');
            }
        });
    }
}