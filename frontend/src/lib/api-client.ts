import type {
  AnyHomeworkResponse,
  EditSubmissionRequest,
  PhotoQuestionSubmission,
  PinUnlockRequest,
} from "./api-types";

/**
 * Base URL for the FastAPI backend (backend/app/main.py, routers mounted under /api).
 * Overridable via NEXT_PUBLIC_API_BASE_URL; defaults to the backend's default
 * `uvicorn app.main:app --reload` address (port 8000).
 */
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/**
 * Distinct failure kinds this client surfaces, so screen components can branch their UI
 * (e.g. network failure vs. wrong PIN vs. lockout) instead of string-matching a message.
 */
export type ApiErrorKind =
  | "network" // fetch threw: server unreachable, offline, DNS or CORS block
  | "not_found" // 404
  | "unprocessable" // 422: named backend guard (e.g. nothing to unlock, flagged row)
  | "unauthorized" // 401: wrong PIN
  | "rate_limited" // 429: PIN lockout
  | "service_restarting" // 503 with Retry-After: model swap restart window (see Issue 366)
  | "unknown"; // any other non-2xx status

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  readonly retryAfterSeconds: number | null;
  readonly detail: string | null;

  constructor(
    kind: ApiErrorKind,
    message: string,
    options?: { status?: number; retryAfterSeconds?: number; detail?: string },
  ) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = options?.status ?? null;
    this.retryAfterSeconds = options?.retryAfterSeconds ?? null;
    this.detail = options?.detail ?? null;
  }
}

function errorKindForStatus(status: number): ApiErrorKind {
  switch (status) {
    case 401:
      return "unauthorized";
    case 404:
      return "not_found";
    case 422:
      return "unprocessable";
    case 429:
      return "rate_limited";
    case 503:
      return "service_restarting";
    default:
      return "unknown";
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        ...(init?.body && !(init.body instanceof FormData)
          ? { "Content-Type": "application/json" }
          : {}),
        ...init?.headers,
      },
    });
  } catch {
    // fetch() rejects only on network failure (server down, offline, DNS, CORS
    // preflight refusal), not on a 4xx/5xx response. This drives the `failed` screen state.
    throw new ApiError(
      "network",
      "Couldn't reach the server. Check your connection and try again.",
    );
  }

  if (!response.ok) {
    let detail: string | null = null;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? null;
    } catch {
      // Non-JSON error body: detail stays null, status and kind still apply.
    }

    if (response.status === 503) {
      const retryAfterHeader = response.headers.get("Retry-After");
      const retryAfterSeconds = retryAfterHeader
        ? Number.parseInt(retryAfterHeader, 10)
        : null;
      throw new ApiError(
        "service_restarting",
        detail ?? "The server is restarting. Please retry shortly.",
        {
          status: response.status,
          detail: detail ?? undefined,
          retryAfterSeconds: retryAfterSeconds ?? undefined,
        },
      );
    }

    throw new ApiError(
      errorKindForStatus(response.status),
      detail ?? `Request failed (${response.status}).`,
      { status: response.status, detail: detail ?? undefined },
    );
  }

  return (await response.json()) as T;
}

/** POST /api/submissions/photo-question: pre-extracted text (the typed-question path). */
export function submitPhotoQuestion(
  body: PhotoQuestionSubmission,
): Promise<AnyHomeworkResponse> {
  return request("/api/submissions/photo-question", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** POST /api/submissions/photo: raw photo upload, transcribed by the VLM. */
export function submitPhoto(file: File | Blob): Promise<AnyHomeworkResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request("/api/submissions/photo", {
    method: "POST",
    body: formData,
  });
}

/**
 * POST /api/submissions/{id}/confirm: the student confirms the extracted text, which runs
 * the gate and reasoning stage. `requestId` is the caller's idempotency key, reused
 * verbatim on retry and never regenerated.
 */
export function confirmSubmission(
  studentSubmissionId: number,
  requestId: string,
): Promise<AnyHomeworkResponse> {
  const params = new URLSearchParams({ request_id: requestId });
  return request(
    `/api/submissions/${studentSubmissionId}/confirm?${params.toString()}`,
    { method: "POST" },
  );
}

/** POST /api/submissions/{id}/edit: student-corrected text, before confirming. */
export function editSubmission(
  studentSubmissionId: number,
  body: EditSubmissionRequest,
): Promise<AnyHomeworkResponse> {
  return request(`/api/submissions/${studentSubmissionId}/edit`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** POST /api/questions/{id}/hint: typed-question direct path (no photo or confirm step). */
export function requestHint(
  questionId: number,
  requestId: string,
): Promise<AnyHomeworkResponse> {
  const params = new URLSearchParams({ request_id: requestId });
  return request(
    `/api/questions/${questionId}/hint?${params.toString()}`,
    { method: "POST" },
  );
}

/** POST /api/attempts/{id}/escalate: "still stuck? get another hint". */
export function escalateHint(
  attemptId: number,
  requestId: string,
): Promise<AnyHomeworkResponse> {
  const params = new URLSearchParams({ request_id: requestId });
  return request(
    `/api/attempts/${attemptId}/escalate?${params.toString()}`,
    { method: "POST" },
  );
}

/**
 * POST /api/questions/{id}/unlock: parent-PIN unlock. Per-question with no session or
 * cache, so every call re-verifies the PIN.
 */
export function unlockSolution(
  questionId: number,
  body: PinUnlockRequest,
): Promise<AnyHomeworkResponse> {
  return request(`/api/questions/${questionId}/unlock`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
