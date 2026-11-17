package com.example.microfly

import android.opengl.Matrix
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import com.google.android.filament.Engine
import io.github.sceneview.Scene
import io.github.sceneview.math.Position
import io.github.sceneview.math.Rotation
import io.github.sceneview.math.Size
import io.github.sceneview.model.ModelInstance
import io.github.sceneview.node.CubeNode
import io.github.sceneview.node.ModelNode
import io.github.sceneview.rememberEngine
import io.github.sceneview.rememberMaterialLoader
import io.github.sceneview.rememberModelLoader
import io.github.sceneview.rememberCameraNode
import io.github.sceneview.rememberMainLightNode
import io.github.sceneview.rememberNodes
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

// ---- world setup (1 unit ~ 1 "fly size") -------------------------------------------------------
private const val BOX_HALF = 4f
private const val FLY_SIZE = 1f
private const val FLY_RADIUS = 0.45f
private const val MODEL_FORWARD_YAW_OFFSET = 0f     // adjust if your GLB faces another axis
private val WING_NAMES = arrayOf("Wing_L", "Wing_R") // node names inside fly.glb (see instructions)

/** Orbit camera: yaw/pitch/distance around a focus point that smoothly follows the fly. */
private class OrbitCamera {
    var yaw = 0.5f; var pitch = 0.45f
    var dist = 7f; private var targetDist = 7f
    private var fx = 0f; private var fy = 0f; private var fz = 0f
    private val camPos = Position(0f, 0f, 0f)

    fun orbit(dx: Float, dy: Float) { yaw -= dx * 0.005f; pitch = (pitch + dy * 0.005f).coerceIn(-1.2f, 1.4f) }
    fun twist(d: Float) { yaw -= d }
    fun zoom(scale: Float) { targetDist = (targetDist / scale).coerceIn(MIN_DIST, MAX_DIST) }

    fun update(dt: Float, tx: Float, ty: Float, tz: Float, cam: io.github.sceneview.node.CameraNode) {
        val s = minOf(1f, dt * 8f)                      // exponential smoothing
        fx += (tx - fx) * s; fy += (ty - fy) * s; fz += (tz - fz) * s
        dist += (targetDist - dist) * s
        val cp = cos(pitch)
        cam.position = Position(fx + dist * cp * sin(yaw), fy + dist * sin(pitch), fz + dist * cp * cos(yaw))
        camPos.x = fx; camPos.y = fy; camPos.z = fz
        cam.lookAt(camPos)                              // focal point = the fly
    }
    companion object { const val MIN_DIST = 0.9f; const val MAX_DIST = 14f }
}

/** Procedural wing flapping: rotates the wing entities' LOCAL transform on top of their rest pose. */
private class WingRig(private val engine: Engine, instance: ModelInstance) {
    private val tm = engine.transformManager
    private val ent = IntArray(2)
    private val ti = IntArray(2)
    private val base = Array(2) { FloatArray(16) }
    private val rot = FloatArray(16)
    private val out = FloatArray(16)
    val valid: Boolean

    init {
        var ok = true
        for (i in 0..1) {
            ent[i] = instance.asset.getFirstEntityByName(WING_NAMES[i])
            if (ent[i] == 0) { ok = false; continue }
            ti[i] = tm.getInstance(ent[i])
            tm.getTransform(ti[i], base[i])
        }
        valid = ok
    }

    /** Flap around the body's longitudinal axis (Z). Change the axis if your model differs. */
    fun apply(angleDeg: Float) {
        if (!valid) return
        for (i in 0..1) {
            val side = if (i == 0) 1f else -1f
            Matrix.setRotateM(rot, 0, angleDeg * side, 0f, 0f, 1f)
            Matrix.multiplyMM(out, 0, base[i], 0, rot, 0)
            tm.setTransform(ti[i], out)
        }
    }
}

@Composable
fun FlyRenderer(modifier: Modifier = Modifier) {
    val engine = rememberEngine()
    val modelLoader = rememberModelLoader(engine)
    val materialLoader = rememberMaterialLoader(engine)
    val cameraNode = rememberCameraNode(engine) { position = Position(0f, 3f, 7f) }
    val mainLight = rememberMainLightNode(engine) { intensity = 110_000f }
    val childNodes = rememberNodes()
    val flyInstance: ModelInstance? = remember(modelLoader) {
        try { modelLoader.createModelInstance("models/fly.glb") } catch (e: Exception) { null }
    }

    val density = LocalDensity.current.density
    val physics = remember { PhysicsManager(BOX_HALF, BOX_HALF, BOX_HALF, FLY_RADIUS) }
    val brain = remember { FlyBehaviorController(physics) }
    val orbit = remember { OrbitCamera() }
    val touch = remember { TouchHandler(density) }
    val lastNanos = remember { LongArray(1) }           // plain array: no recomposition per frame
    val yawState = remember { floatArrayOf(0f, 0f) }    // [smoothed yaw, wing phase]

    // Glass slide under the fly (dark, matte) to give the microscope-stage feel.
    val floor = remember(engine) {
        CubeNode(
            engine,
            size = Size(BOX_HALF * 2.4f, 0.1f, BOX_HALF * 2.4f),
            center = Position(0f, 0f, 0f),
            materialInstance = materialLoader.createColorInstance(Color(0xFF1B2A30), metallic = 0f, roughness = 0.15f, reflectance = 0.8f)
        ).apply { position = Position(0f, -BOX_HALF - 0.1f, 0f) }
    }
    val flyNode = remember(flyInstance) {
        flyInstance?.let { ModelNode(modelInstance = it, autoAnimate = false, scaleToUnits = FLY_SIZE) }
    }
    val wings = remember(flyInstance) { flyInstance?.let { WingRig(engine, it) } }

    LaunchedEffect(floor) { childNodes.add(floor) }
    LaunchedEffect(flyNode) { flyNode?.let { childNodes.add(it) } }

    // Wire touch -> behaviour/camera
    remember(touch) {
        touch.listener = object : TouchHandler.Listener {
            override fun onFlyTapped(awayX: Float, awayY: Float, awayZ: Float) = brain.startle(awayX, awayY, awayZ)
            override fun onFlyDragged(awayX: Float, awayY: Float, awayZ: Float) = brain.panic(awayX, awayY, awayZ)
            override fun onOrbit(dxPx: Float, dyPx: Float) = orbit.orbit(dxPx, dyPx)
            override fun onPinch(scale: Float) = orbit.zoom(scale)
            override fun onTwist(deltaRad: Float) = orbit.twist(deltaRad)
        }
        true
    }

    Box(modifier = modifier.flyTouch(touch)) {
        Scene(
            modifier = Modifier.fillMaxSize(),
            engine = engine,
            modelLoader = modelLoader,
            cameraNode = cameraNode,
            childNodes = childNodes,
            mainLightNode = mainLight,
            cameraManipulator = null,                   // we run our own orbit camera
            onFrame = { frameNanos ->
                val dt = if (lastNanos[0] == 0L) 0.016f
                         else ((frameNanos - lastNanos[0]) * 1e-9f).coerceIn(0.001f, 0.05f)
                lastNanos[0] = frameNanos

                // 1) AI decides accelerations, 2) physics integrates + collides
                brain.update(dt)
                physics.step(dt)

                // 3) Fly pose is in WORLD space -> unaffected by camera orbit.
                val speed2 = physics.vx * physics.vx + physics.vz * physics.vz
                if (speed2 > 0.04f) {
                    val target = Math.toDegrees(atan2(physics.vx, physics.vz).toDouble()).toFloat() + MODEL_FORWARD_YAW_OFFSET
                    var d = (target - yawState[0]) % 360f
                    if (d > 180f) d -= 360f; if (d < -180f) d += 360f
                    yawState[0] += d * minOf(1f, dt * 10f)
                }
                val pitchDeg = -Math.toDegrees(
                    atan2(physics.vy.toDouble(), sqrt(speed2.toDouble()).coerceAtLeast(0.2))
                ).toFloat() * 0.5f
                flyNode?.let {
                    it.position = Position(physics.px, physics.py, physics.pz)
                    it.rotation = Rotation(pitchDeg, yawState[0], 0f)
                }

                // 4) Wings flap independently of body motion (procedural).
                yawState[1] += (2f * Math.PI.toFloat()) * brain.wingHz * dt
                wings?.apply(brain.wingAmp * sin(yawState[1]))

                // 5) Camera + picking data for the ray caster.
                touch.camera = cameraNode.camera
                touch.flyX = physics.px; touch.flyY = physics.py; touch.flyZ = physics.pz
                touch.flyRadius = FLY_RADIUS
                orbit.update(dt, physics.px, physics.py, physics.pz, cameraNode)
            }
        )

        // Microscope eyepiece vignette (cheap 2D overlay, no touch handling).
        Canvas(Modifier.fillMaxSize()) {
            val r = size.maxDimension * 0.55f
            drawRect(
                Brush.radialGradient(
                    0.55f to Color.Transparent, 1f to Color(0xEE000000),
                    center = Offset(size.width / 2f, size.height / 2f), radius = r
                )
            )
        }
    }
}
