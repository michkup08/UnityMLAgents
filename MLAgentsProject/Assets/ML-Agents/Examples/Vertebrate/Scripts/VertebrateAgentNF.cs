using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

[RequireComponent(typeof(JointDriveController))] 
public class VertebrateAgentNF : Agent
{

    [Header("Walk Speed")]
    [Range(0.1f, m_maxWalkingSpeed)]
    [SerializeField]
    
    private float m_TargetWalkingSpeed = m_maxWalkingSpeed;

    const float m_maxWalkingSpeed = 15;

    public float TargetWalkingSpeed
    {
        get { return m_TargetWalkingSpeed; }
        set { m_TargetWalkingSpeed = Mathf.Clamp(value, .1f, m_maxWalkingSpeed); }
    }

    [Header("Target To Walk Towards")]
    public Transform TargetPrefab; 
    private Transform m_Target; 

    [Header("Body Parts")][Space(10)] public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;

    public Transform segment0;
    public Transform segment1;
    public Transform segment2;


    
    OrientationCubeController m_OrientationCube;

    DirectionIndicator m_DirectionIndicator;
    JointDriveController m_JdController;

    [Header("Foot Grounded Visualization")]
    [Space(10)]
    public bool useFootGroundedVisualization;

    public MeshRenderer foot0;
    public MeshRenderer foot1;
    public MeshRenderer foot2;
    public MeshRenderer foot3;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    [Header("Leg Raycast Settings")]
    [Space(10)]
    [Tooltip("Warstwa, na której znajduje się ziemia. Ustaw to tak, aby raycast nie uderzał w same nogi agenta!")]
    public LayerMask groundLayer = ~0;
    public float legRaycastDistance = 3.0f; 

    public TerrainWithMaterial terrainWithMaterial;

    public override void Initialize()
    {
        SpawnTarget(TargetPrefab, transform.position); 

        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();
        m_JdController = GetComponent<JointDriveController>();

        //Setup each body part
        m_JdController.SetupBodyPart(body);
        m_JdController.SetupBodyPart(leg0Upper);
        m_JdController.SetupBodyPart(leg0Lower);
        m_JdController.SetupBodyPart(leg1Upper);
        m_JdController.SetupBodyPart(leg1Lower);
        m_JdController.SetupBodyPart(leg2Upper);
        m_JdController.SetupBodyPart(leg2Lower);
        m_JdController.SetupBodyPart(leg3Upper);
        m_JdController.SetupBodyPart(leg3Lower);

        m_JdController.SetupBodyPart(segment0);
        m_JdController.SetupBodyPart(segment1);
        m_JdController.SetupBodyPart(segment2);
    }

    void SpawnTarget(Transform prefab, Vector3 pos)
    {
        m_Target = Instantiate(prefab, pos, Quaternion.identity, transform.parent);
    }

    public override void OnEpisodeBegin()
    {
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        body.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        UpdateOrientationObjects();

        TargetWalkingSpeed = 10;

        terrainWithMaterial.generateRandomTerrain();
    }

    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        sensor.AddObservation(bp.groundContact.touchingGround);

        if (bp.rb.transform != body)
        {
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        var cubeForward = m_OrientationCube.transform.forward;

        var velGoal = cubeForward * TargetWalkingSpeed;
        var avgVel = GetAvgVelocity();

        sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));
        sensor.AddObservation(Quaternion.FromToRotation(body.forward, cubeForward));

        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(m_Target.transform.position));

        RaycastHit hit;
        float maxRaycastDist = 10;
        if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist, groundLayer))
        {
            sensor.AddObservation(hit.distance / maxRaycastDist);
        }
        else
            sensor.AddObservation(1);

        CollectLegTerrainObservation(leg0Lower, sensor);
        CollectLegTerrainObservation(leg1Lower, sensor);
        CollectLegTerrainObservation(leg2Lower, sensor);
        CollectLegTerrainObservation(leg3Lower, sensor);

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    private void CollectLegTerrainObservation(Transform legTransform, VectorSensor sensor)
    {
        RaycastHit hit;
        if (Physics.Raycast(legTransform.position, Vector3.down, out hit, legRaycastDistance, groundLayer))
        {
            sensor.AddObservation(hit.distance / legRaycastDistance);

            
            Vector3 localSurfaceNormal = m_OrientationCube.transform.InverseTransformDirection(hit.normal);
            sensor.AddObservation(localSurfaceNormal);
        }
        else
        {
            sensor.AddObservation(1f);
            sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(Vector3.up));
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var bpDict = m_JdController.bodyPartsDict;

        var continuousActions = actionBuffers.ContinuousActions;
        var i = -1;
        bpDict[leg0Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg1Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg2Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg3Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg0Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg1Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg2Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg3Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);

        bpDict[segment0].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[segment1].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[segment2].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);


        bpDict[leg0Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg0Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Lower].SetJointStrength(continuousActions[++i]);
        bpDict[segment0].SetJointStrength(continuousActions[++i]);
        bpDict[segment1].SetJointStrength(continuousActions[++i]);
        bpDict[segment2].SetJointStrength(continuousActions[++i]);

    }

    void FixedUpdate()
    {
        UpdateOrientationObjects();

        if (useFootGroundedVisualization)
        {
            foot0.material = m_JdController.bodyPartsDict[leg0Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot1.material = m_JdController.bodyPartsDict[leg1Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot2.material = m_JdController.bodyPartsDict[leg2Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot3.material = m_JdController.bodyPartsDict[leg3Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
        }

        var cubeForward = m_OrientationCube.transform.forward;

        var matchSpeedReward = GetMatchingVelocityReward(cubeForward * TargetWalkingSpeed, GetAvgVelocity());

        var lookAtTargetReward = Mathf.Pow((Vector3.Dot(cubeForward, body.forward) + 1) * .5F, 2); 


        AddReward(matchSpeedReward * lookAtTargetReward);

    }

    void UpdateOrientationObjects()
    {
        m_OrientationCube.UpdateOrientation(segment0, m_Target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }

    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;
        Vector3 avgVel = Vector3.zero;

        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.linearVelocity;
        }

        avgVel = velSum / numOfRb;
        return avgVel;
    }
    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetWalkingSpeed);

        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / TargetWalkingSpeed, 2), 2);
    }

    public void TouchedTarget()
    {
        AddReward(1f);
    }
}
