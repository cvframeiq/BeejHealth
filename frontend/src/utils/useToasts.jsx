import { useState, useCallback } from 'react';

export function useToasts() {
  const [toasts, setToasts] = useState([]);
  const add = useCallback((msg, type = 'ok') => {
    const id = Date.now();
    setToasts(p => [...p, {
      id,
      msg,
      type
    }]);
    setTimeout(() => setToasts(p => p.filter(t => t.id !== id)), 3800);
  }, []);
  return {
    toasts,
    add
  };
}
