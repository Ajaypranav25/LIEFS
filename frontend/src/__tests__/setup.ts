import '@testing-library/jest-dom';

// Polyfill window.scrollTo and crypto.randomUUID
if (!window.scrollTo) {
  window.scrollTo = () => {};
}

if (!window.HTMLElement.prototype.scrollIntoView) {
  window.HTMLElement.prototype.scrollIntoView = () => {};
}

// Mock ResizeObserver for Recharts
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
