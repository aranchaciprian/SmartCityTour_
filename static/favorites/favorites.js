window.toggleFavoriteById = async function(arg){
  let payload = {};
  if (typeof arg === 'number' || typeof arg === 'string') payload = { tui_id: arg };
  else if (arg && typeof arg === 'object') payload = arg;

  const res = await fetch("/api/favorites/toggle", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    credentials: "same-origin",
    redirect: "manual",
    body: JSON.stringify(payload)
  });
  if (res.status === 401 || res.redirected || res.type === "opaqueredirect") {
    return { ok:false, error:"AUTH_REQUIRED" };
  }
  try { return await res.json(); } catch { return { ok:false, error:"BAD_RESPONSE" }; }
};

window.loadMyFavorites = async function(){
  const res = await fetch("/api/favorites", { credentials:"same-origin", redirect:"manual" });
  if (res.status === 401 || res.redirected || res.type === "opaqueredirect") return { tui_ids: [], place_ids: [] };
  const data = await res.json().catch(()=>({}));
  return { tui_ids: data.tui_ids || [], place_ids: data.place_ids || [] };
};
