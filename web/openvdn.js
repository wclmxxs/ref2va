import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
  name: "openvdn.h200.preview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["OpenVDNH200Generate", "OpenVDNH200Request"].includes(nodeData.name)) return;
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      executed?.apply(this, arguments);
      const item = message.openvdn_videos?.[0];
      if (!item) return;
      if (!this.openvdnPlayer) {
        const video = document.createElement("video");
        video.controls = true;
        video.style.width = "100%";
        this.addDOMWidget("preview", "video", video, { serialize: false });
        this.openvdnPlayer = video;
      }
      this.openvdnPlayer.src = api.apiURL("/view?" + new URLSearchParams(item));
    };
  }
});
