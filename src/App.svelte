<style lang="sass">
    \:global(*)
        box-sizing: border-box

    \:global(html)
        scrollbar-color: black lightgray
            
    \:global(body)
        font-family: "Fira Sans", "Noto Sans", "Segoe UI", Verdana, sans-serif
        font-weight: 300
        overflow-anchor: none
        //margin: 0
        //min-height: 100vh

    \:global(strong)
        font-weight: bold

    @mixin header
        border-bottom: 1px solid gray
        margin: 0
        margin-bottom: 0.5em
        font-weight: 500
        //a
            //color: inherit

    \:global(h1)
        @include header
    \:global(h2)
        @include header
    \:global(h3)
        @include header
    \:global(h4)
        @include header
    \:global(h5)
        @include header
    \:global(h6)
        @include header
    \:global(ul)
        list-style-type: square
        padding: 0
        padding-left: 1em

    input, button, select
        border-radius: 0
        border: 1px solid gray
        padding: 0.5em

    .game-display
        .row
            white-space: nowrap
            .cell
                display: inline-block
                text-align: center

    .wrapper
        display: flex
</style>

<h1>EWO3 Memetic Edition</h1>

<div class="wrapper">
    <div class="game-display">
        {#each grid as row, y}
            <div class="row" style={`height: ${VERT}px; ` + (y % 2 === 1 ? `padding-left: ${HORIZ/2}px` : "")}>
                {#each row as cell}
                    <div class="cell" style={`width: ${HORIZ}px; height: ${VERT}px; line-height: ${VERT}px; opacity: ${cell[1] * 100}%`}>{cell[0]}</div>
                {/each}
            </div>
        {/each}
    </div>
    <div class="controls">
        {#if dead}
            You have died to death. <a href="#" on:click={restart}>Restart</a>.
        {/if}
        {#if players}
            {players} connected players.
        {/if}
        {#if health}
            Your health is {health}.
        {/if}
        <ul>
            {#each inventory as item}
                <li>{item[0]} x{item[2]}: {item[1]}</li>
            {/each}
        </ul>
    </div>
</div>

<svelte:window on:keydown={keydown} on:keyup={keyup}></svelte:window>

<script>
    import * as util from "./util"

    let dead = false
    let health
    let players
    let inventory = []

    let ws
    const connect = () => {
        ws = new WebSocket(window.location.protocol === "https:" ? "wss://ewo.osmarks.net/" : "ws://localhost:8080/")

        ws.addEventListener("message", ev => {
            const data = JSON.parse(ev.data)
            if (data.Display) {
                const newGrid = blankGrid()
                for (const [q, r, c, o] of data.Display.nearby) {
                    const col = q + (r - (r & 1)) / 2
                    const row = r
                    newGrid[row + OFFSET][col + OFFSET] = [c, o]
                }
                grid = newGrid
                health = data.Display.health
                inventory = data.Display.inventory
            }
            if (data === "Dead") {
                dead = true
            }
            if (data.PlayerCount) {
                players = data.PlayerCount
            }
            for (const key of keysDown) {
                const input = INPUTS[key]
                if (input) {
                    ws.send(JSON.stringify(input))
                }
            }
            keysDown = new Set(Array.from(keysDown).map(k => !keysCleared.has(k)))
            keysCleared = new Set()
        })

        ws.addEventListener("close", ev => {
            console.warn("oh no")
        })
    }

    const reconnect = () => {
        if (ws) ws.close()
        connect()
    }

    const restart = ev => {
        ev.preventDefault()
        dead = false
        reconnect()
    }

    const GRIDSIZE = 33
    const OFFSET = Math.floor(GRIDSIZE/2)
    const SIZE = 16
    const HORIZ = Math.sqrt(3) * SIZE
    const VERT = 3/2 * SIZE

    const blankGrid = () => new Array(GRIDSIZE).fill(null).map(() => new Array(GRIDSIZE).fill("​"))

    let grid = blankGrid()

    let keysDown = new Set()
    let keysCleared = new Set()

    const keydown = ev => {
        keysDown.add(ev.key)
    }
    const keyup = ev => {
        keysCleared.add(ev.key)
    }

    const INPUTS = {
        "w": "UpLeft",
        "e": "UpRight",
        "a": "Left",
        "d": "Right",
        "z": "DownLeft",
        "x": "DownRight",
        "f": "Dig"
    }

    connect()
</script>
