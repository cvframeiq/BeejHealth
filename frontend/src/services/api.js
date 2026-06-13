export const API = {
  BASE: "",   // Vite proxy handles /api → localhost:3000 in dev
  _headers(isJson = false) {
    const token = localStorage.getItem("bh_token");
    const h = {};
    if (isJson) h["Content-Type"] = "application/json";
    if (token)  h["Authorization"] = "Bearer " + token;
    return h;
  },
  async _fetch(method, path, body, timeoutMs = 12000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const opts = { method, headers: this._headers(!!body), signal: controller.signal };
    if (body) opts.body = JSON.stringify(body);
    try {
      const r = await fetch(this.BASE + path, opts);
      const raw = await r.text();
      let data = {};
      try {
        data = raw ? JSON.parse(raw) : {};
      } catch {
        data = { error: raw || "Server error" };
      }
      if (!r.ok) throw new Error(data.error || "Server error");
      return data;
    } catch (err) {
      if (err?.name === "AbortError") {
        throw new Error("Server response slow hai. Dobara try karein.");
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  },
  get:    (path)               => API._fetch("GET",    path),
  post:   (path, body, timeoutMs) => API._fetch("POST",   path, body, timeoutMs),
  patch:  (path, body)         => API._fetch("PATCH",  path, body),
  delete: (path)               => API._fetch("DELETE", path),
};

export const MIN_REPORT_PHOTOS = 3;
export const MAX_REPORT_PHOTOS = 5;
export const MAX_UPLOAD_SOURCE_BYTES = 25 * 1024 * 1024;
export const MAX_UPLOAD_PAYLOAD_BYTES = 34 * 1024 * 1024;
export const MAX_UPLOAD_DIMENSION = 1600;
export const UPLOAD_JPEG_QUALITY = 0.84;

export function saveSession(token, user) {
  if (!token || !user) return;
  localStorage.setItem("bh_token", token);
  localStorage.setItem("bh_user", JSON.stringify(user));
}
export function clearSession() {
  localStorage.removeItem("bh_token");
  localStorage.removeItem("bh_user");
}
export function loadSession() {
  try {
    const token = localStorage.getItem("bh_token");
    const raw   = localStorage.getItem("bh_user");
    const user  = raw ? JSON.parse(raw) : null;
    if (user && !user._id && user.id) user._id = user.id;
    if (user && !user.id && user._id) user.id = user._id;
    if (!token && user) {
      localStorage.removeItem("bh_token");
      localStorage.removeItem("bh_user");
      return { token: null, user: null };
    }
    // Validate user object has required fields
    if (user && (!(user?._id || user?.id) || !user?.name || !user?.type)) {
      // Stale/corrupt user data — clear it
      localStorage.removeItem("bh_token");
      localStorage.removeItem("bh_user");
      return { token: null, user: null };
    }
    return { token, user };
  } catch {
    localStorage.removeItem("bh_token");
    localStorage.removeItem("bh_user");
    return { token: null, user: null };
  }
}
