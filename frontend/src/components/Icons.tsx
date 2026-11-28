// Small inline icons (no icon library needed). All use currentColor.
import type { ReactNode } from "react";

type P = { size?: number };

const svg = (size: number, children: ReactNode) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.8}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    {children}
  </svg>
);

export const LungsIcon = ({ size = 22 }: P) =>
  svg(
    size,
    <>
      <path d="M12 3v8" />
      <path d="M12 11c-1.5 1-3 1.5-3 4" />
      <path d="M12 11c1.5 1 3 1.5 3 4" />
      <path d="M8.5 6.5C5 7 3 12 3 17c0 2 1.2 3 3 3 2.5 0 3.5-2 3.5-5V8c0-1-.4-1.6-1-1.5Z" />
      <path d="M15.5 6.5C19 7 21 12 21 17c0 2-1.2 3-3 3-2.5 0-3.5-2-3.5-5V8c0-1 .4-1.6 1-1.5Z" />
    </>,
  );

export const UploadIcon = ({ size = 18 }: P) =>
  svg(
    size,
    <>
      <path d="M12 15V4" />
      <path d="m7 9 5-5 5 5" />
      <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
    </>,
  );

export const LayersIcon = ({ size = 16 }: P) =>
  svg(
    size,
    <>
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m3 13 9 5 9-5" />
    </>,
  );

export const ClockIcon = ({ size = 14 }: P) =>
  svg(
    size,
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </>,
  );

export const AlertIcon = ({ size = 14 }: P) =>
  svg(
    size,
    <>
      <path d="M12 9v4" />
      <path d="M12 17h.01" />
      <path d="M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
    </>,
  );

export const GridIcon = ({ size = 16 }: P) =>
  svg(
    size,
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </>,
  );
