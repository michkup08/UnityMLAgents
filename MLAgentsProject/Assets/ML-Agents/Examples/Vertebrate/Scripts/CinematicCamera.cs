using UnityEngine;

public class CinematicCamera : MonoBehaviour
{
    [Header("Ustawienia Ruchu (Translacja)")]
    public float moveSpeed = 10f;
    [Tooltip("Płynność hamowania i podążania za celem.")]
    public float moveSmoothTime = 0.3f;
    [Tooltip("Czas rozpędzania się z miejsca (miękki start).")]
    public float movementAccelerationTime = 0.5f;

    [Header("Ustawienia Obrotu (Mysz)")]
    public float rotationSpeed = 3f;
    [Tooltip("Płynność zatrzymywania obrotu.")]
    public float rotationSmoothTime = 0.15f;
    [Tooltip("Czas rozpędzania obrotu (powolny start ruchu myszką).")]
    public float mouseAccelerationTime = 0.1f;

    [Header("Dodatkowe")]
    public KeyCode upKey = KeyCode.E;
    public KeyCode downKey = KeyCode.Q;

    private Vector3 targetPosition;
    private float targetYaw;
    private float targetPitch;

    private float currentYaw;
    private float currentPitch;
    private Vector3 positionVelocity;
    private float yawVelocity;
    private float pitchVelocity;

    private Vector3 smoothedInputVector;
    private Vector3 inputVectorVelocity;
    private float smoothedMouseX;
    private float smoothedMouseY;
    private float mouseXVelocity;
    private float mouseYVelocity;

    void Start()
    {
        targetPosition = transform.position;

        Vector3 angles = transform.eulerAngles;
        targetPitch = angles.x;
        targetYaw = angles.y;
        currentPitch = targetPitch;
        currentYaw = targetYaw;

        LockCursor(true);
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape)) LockCursor(false);
        if (Input.GetMouseButtonDown(0)) LockCursor(true);

        HandleInput();
    }

    void LateUpdate()
    {
        ApplySmoothing();
    }

    private void LockCursor(bool state)
    {
        Cursor.lockState = state ? CursorLockMode.Locked : CursorLockMode.None;
        Cursor.visible = !state;
    }

    private void HandleInput()
    {
        float rawMouseX = 0f;
        float rawMouseY = 0f;

        if (Cursor.lockState == CursorLockMode.Locked)
        {
            rawMouseX = Input.GetAxisRaw("Mouse X") * rotationSpeed;
            rawMouseY = Input.GetAxisRaw("Mouse Y") * rotationSpeed;
        }

        smoothedMouseX = Mathf.SmoothDamp(smoothedMouseX, rawMouseX, ref mouseXVelocity, mouseAccelerationTime);
        smoothedMouseY = Mathf.SmoothDamp(smoothedMouseY, rawMouseY, ref mouseYVelocity, mouseAccelerationTime);

        targetYaw += smoothedMouseX;
        targetPitch -= smoothedMouseY;
        targetPitch = Mathf.Clamp(targetPitch, -89f, 89f);

        float rawMoveX = Input.GetAxisRaw("Horizontal");
        float rawMoveZ = Input.GetAxisRaw("Vertical");
        float rawMoveY = 0f;

        if (Input.GetKey(upKey)) rawMoveY += 1f;
        if (Input.GetKey(downKey)) rawMoveY -= 1f;

        Quaternion yawRotation = Quaternion.Euler(0f, targetYaw, 0f);
        Vector3 flatDirection = new Vector3(rawMoveX, 0f, rawMoveZ).normalized;
        Vector3 targetInputVector = (yawRotation * flatDirection) + new Vector3(0f, rawMoveY, 0f);

        if (targetInputVector.sqrMagnitude > 1f)
        {
            targetInputVector.Normalize();
        }

        smoothedInputVector = Vector3.SmoothDamp(smoothedInputVector, targetInputVector, ref inputVectorVelocity, movementAccelerationTime);

        float currentSpeed = Input.GetKey(KeyCode.LeftShift) ? moveSpeed * 2f : moveSpeed;

        // Zmiana pozycji celu
        targetPosition += smoothedInputVector * currentSpeed * Time.deltaTime;
    }

    private void ApplySmoothing()
    {
        currentYaw = Mathf.SmoothDamp(currentYaw, targetYaw, ref yawVelocity, rotationSmoothTime);
        currentPitch = Mathf.SmoothDamp(currentPitch, targetPitch, ref pitchVelocity, rotationSmoothTime);

        transform.rotation = Quaternion.Euler(currentPitch, currentYaw, 0f);

        transform.position = Vector3.SmoothDamp(transform.position, targetPosition, ref positionVelocity, moveSmoothTime);
    }
}
