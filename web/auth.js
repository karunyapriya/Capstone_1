// AgriSense - Supabase auth wrapper.
// Supabase authenticates the user; it never sees an image and never makes a
// prediction. This module is the only thing in web/ that talks to Supabase.

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import { SUPABASE_URL, SUPABASE_ANON_KEY } from './config.js';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

/**
 * @returns {Promise<{ok:true, needsConfirmation:boolean}|{ok:false, error:string}>}
 */
export async function signUp(email, password) {
  const { data, error } = await supabase.auth.signUp({ email, password });
  if (error) return { ok: false, error: error.message };
  // If email confirmation is required project-side, signUp succeeds but
  // returns no session - the caller can't tell the difference from the error
  // alone, so check for a session explicitly.
  return { ok: true, needsConfirmation: !data.session };
}

/**
 * @returns {Promise<{ok:true}|{ok:false, error:string}>}
 */
export async function signInWithPassword(email, password) {
  const { error } = await supabase.auth.signInWithPassword({ email, password });
  if (error) return { ok: false, error: error.message };
  return { ok: true };
}

export async function signOut() {
  await supabase.auth.signOut();
}

/** @returns {Promise<import('@supabase/supabase-js').Session|null>} */
export async function getSession() {
  const { data } = await supabase.auth.getSession();
  return data.session;
}

/** @param {(session: import('@supabase/supabase-js').Session|null) => void} cb */
export function onAuthStateChange(cb) {
  const { data } = supabase.auth.onAuthStateChange((_event, session) => cb(session));
  return data.subscription;
}
