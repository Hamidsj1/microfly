package com.example.microfly

import android.content.Context
import android.media.AudioAttributes
import android.media.SoundPool

/**
 * Procedural fly sounds (res/raw, made by tools/make_fly_sounds.py).
 * Two looping buzz streams (hover / fast flight) are cross-faded by the fly's speed and their
 * playback rate follows the speed; one-shots fire on state changes.
 */
class FlySound(context: Context) {

    private val pool: SoundPool = SoundPool.Builder()
        .setMaxStreams(8)
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_GAME)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build()
        ).build()

    private val hoverId = pool.load(context, R.raw.buzz_hover, 1)
    private val flyId = pool.load(context, R.raw.buzz_fly, 1)
    private val walkId = pool.load(context, R.raw.walk_steps, 1)
    private val groomId = pool.load(context, R.raw.groom, 1)
    private val dartId = pool.load(context, R.raw.buzz_dart, 1)
    private val takeoffId = pool.load(context, R.raw.takeoff, 1)
    private val landingId = pool.load(context, R.raw.landing, 1)

    private val loaded = HashSet<Int>()
    private var hoverS = 0
    private var flyS = 0
    private var walkS = 0
    private var groomS = 0

    private var vHover = 0f
    private var vFly = 0f
    private var vWalk = 0f
    private var vGroom = 0f
    private var flyness = 0f
    private var prev = FlyState.IDLE
    private var released = false

    init {
        pool.setOnLoadCompleteListener { _, id, status ->
            if (status == 0) synchronized(loaded) { loaded.add(id) }
        }
        current = this
    }

    private fun ready(id: Int): Boolean = synchronized(loaded) { id in loaded }

    private fun startLoop(id: Int): Int =
        if (ready(id)) pool.play(id, 0f, 0f, 1, -1, 1f) else 0

    fun update(state: FlyState, speed: Float, dt: Float) {
        if (released) return
        // --- one-shots on state change ---
        if (state != prev) {
            when {
                state == FlyState.STARTLE && prev != FlyState.EVADE -> shot(dartId, 0.8f)
                state == FlyState.CLING -> shot(landingId, 0.7f)
                prev == FlyState.CLING -> shot(takeoffId, 0.7f)
            }
            prev = state
        }
        // --- loop streams (started lazily once the sample is decoded) ---
        if (hoverS == 0) hoverS = startLoop(hoverId)
        if (flyS == 0) flyS = startLoop(flyId)
        if (walkS == 0) walkS = startLoop(walkId)
        if (groomS == 0) groomS = startLoop(groomId)

        val k = minOf(1f, dt * 10f)
        val targetFly = ((speed - 0.4f) / 5f).coerceIn(0f, 1f)
        flyness += (targetFly - flyness) * k
        val clinging = state == FlyState.CLING
        val tHover = if (clinging) 0f else (1f - flyness) * 0.35f
        val tFly = if (clinging) 0f else flyness * 0.75f
        val tWalk = if (clinging) 0.5f else 0f
        val tGroom = if (state == FlyState.GROOM) 0.6f else 0f
        vHover += (tHover - vHover) * k
        vFly += (tFly - vFly) * k
        vWalk += (tWalk - vWalk) * k
        vGroom += (tGroom - vGroom) * k

        if (hoverS != 0) pool.setVolume(hoverS, vHover, vHover)
        if (flyS != 0) {
            pool.setVolume(flyS, vFly, vFly)
            pool.setRate(flyS, (0.9f + 0.4f * flyness).coerceIn(0.5f, 2f))   // pitch follows speed
        }
        if (walkS != 0) pool.setVolume(walkS, vWalk, vWalk)
        if (groomS != 0) pool.setVolume(groomS, vGroom, vGroom)
    }

    private fun shot(id: Int, vol: Float) {
        if (ready(id)) pool.play(id, vol, vol, 2, 0, 1f)
    }

    fun pause() {
        if (!released) pool.autoPause()
    }

    fun resume() {
        if (!released) pool.autoResume()
    }

    fun release() {
        if (released) return
        released = true
        pool.release()
        if (current === this) current = null
    }

    companion object {
        @Volatile
        var current: FlySound? = null
    }
}
