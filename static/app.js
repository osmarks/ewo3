(() => {
  // node_modules/svelte/internal/index.mjs
  function noop() {
  }
  function run(fn) {
    return fn();
  }
  function blank_object() {
    return Object.create(null);
  }
  function run_all(fns) {
    fns.forEach(run);
  }
  function is_function(thing) {
    return typeof thing === "function";
  }
  function safe_not_equal(a, b) {
    return a != a ? b == b : a !== b || (a && typeof a === "object" || typeof a === "function");
  }
  function is_empty(obj) {
    return Object.keys(obj).length === 0;
  }
  var tasks = new Set();
  var is_hydrating = false;
  function start_hydrating() {
    is_hydrating = true;
  }
  function end_hydrating() {
    is_hydrating = false;
  }
  function upper_bound(low, high, key, value) {
    while (low < high) {
      const mid = low + (high - low >> 1);
      if (key(mid) <= value) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    return low;
  }
  function init_hydrate(target) {
    if (target.hydrate_init)
      return;
    target.hydrate_init = true;
    const children2 = target.childNodes;
    const m = new Int32Array(children2.length + 1);
    const p = new Int32Array(children2.length);
    m[0] = -1;
    let longest = 0;
    for (let i = 0; i < children2.length; i++) {
      const current = children2[i].claim_order;
      const seqLen = upper_bound(1, longest + 1, (idx) => children2[m[idx]].claim_order, current) - 1;
      p[i] = m[seqLen] + 1;
      const newLen = seqLen + 1;
      m[newLen] = i;
      longest = Math.max(newLen, longest);
    }
    const lis = [];
    const toMove = [];
    let last = children2.length - 1;
    for (let cur = m[longest] + 1; cur != 0; cur = p[cur - 1]) {
      lis.push(children2[cur - 1]);
      for (; last >= cur; last--) {
        toMove.push(children2[last]);
      }
      last--;
    }
    for (; last >= 0; last--) {
      toMove.push(children2[last]);
    }
    lis.reverse();
    toMove.sort((a, b) => a.claim_order - b.claim_order);
    for (let i = 0, j = 0; i < toMove.length; i++) {
      while (j < lis.length && toMove[i].claim_order >= lis[j].claim_order) {
        j++;
      }
      const anchor = j < lis.length ? lis[j] : null;
      target.insertBefore(toMove[i], anchor);
    }
  }
  function append(target, node) {
    if (is_hydrating) {
      init_hydrate(target);
      if (target.actual_end_child === void 0 || target.actual_end_child !== null && target.actual_end_child.parentElement !== target) {
        target.actual_end_child = target.firstChild;
      }
      if (node !== target.actual_end_child) {
        target.insertBefore(node, target.actual_end_child);
      } else {
        target.actual_end_child = node.nextSibling;
      }
    } else if (node.parentNode !== target) {
      target.appendChild(node);
    }
  }
  function insert(target, node, anchor) {
    if (is_hydrating && !anchor) {
      append(target, node);
    } else if (node.parentNode !== target || anchor && node.nextSibling !== anchor) {
      target.insertBefore(node, anchor || null);
    }
  }
  function detach(node) {
    node.parentNode.removeChild(node);
  }
  function destroy_each(iterations, detaching) {
    for (let i = 0; i < iterations.length; i += 1) {
      if (iterations[i])
        iterations[i].d(detaching);
    }
  }
  function element(name) {
    return document.createElement(name);
  }
  function text(data) {
    return document.createTextNode(data);
  }
  function space() {
    return text(" ");
  }
  function listen(node, event, handler, options) {
    node.addEventListener(event, handler, options);
    return () => node.removeEventListener(event, handler, options);
  }
  function attr(node, attribute, value) {
    if (value == null)
      node.removeAttribute(attribute);
    else if (node.getAttribute(attribute) !== value)
      node.setAttribute(attribute, value);
  }
  function children(element2) {
    return Array.from(element2.childNodes);
  }
  function set_data(text2, data) {
    data = "" + data;
    if (text2.wholeText !== data)
      text2.data = data;
  }
  var active_docs = new Set();
  var current_component;
  function set_current_component(component) {
    current_component = component;
  }
  var dirty_components = [];
  var binding_callbacks = [];
  var render_callbacks = [];
  var flush_callbacks = [];
  var resolved_promise = Promise.resolve();
  var update_scheduled = false;
  function schedule_update() {
    if (!update_scheduled) {
      update_scheduled = true;
      resolved_promise.then(flush);
    }
  }
  function add_render_callback(fn) {
    render_callbacks.push(fn);
  }
  var flushing = false;
  var seen_callbacks = new Set();
  function flush() {
    if (flushing)
      return;
    flushing = true;
    do {
      for (let i = 0; i < dirty_components.length; i += 1) {
        const component = dirty_components[i];
        set_current_component(component);
        update(component.$$);
      }
      set_current_component(null);
      dirty_components.length = 0;
      while (binding_callbacks.length)
        binding_callbacks.pop()();
      for (let i = 0; i < render_callbacks.length; i += 1) {
        const callback = render_callbacks[i];
        if (!seen_callbacks.has(callback)) {
          seen_callbacks.add(callback);
          callback();
        }
      }
      render_callbacks.length = 0;
    } while (dirty_components.length);
    while (flush_callbacks.length) {
      flush_callbacks.pop()();
    }
    update_scheduled = false;
    flushing = false;
    seen_callbacks.clear();
  }
  function update($$) {
    if ($$.fragment !== null) {
      $$.update();
      run_all($$.before_update);
      const dirty = $$.dirty;
      $$.dirty = [-1];
      $$.fragment && $$.fragment.p($$.ctx, dirty);
      $$.after_update.forEach(add_render_callback);
    }
  }
  var outroing = new Set();
  function transition_in(block, local) {
    if (block && block.i) {
      outroing.delete(block);
      block.i(local);
    }
  }
  var globals = typeof window !== "undefined" ? window : typeof globalThis !== "undefined" ? globalThis : global;
  var boolean_attributes = new Set([
    "allowfullscreen",
    "allowpaymentrequest",
    "async",
    "autofocus",
    "autoplay",
    "checked",
    "controls",
    "default",
    "defer",
    "disabled",
    "formnovalidate",
    "hidden",
    "ismap",
    "loop",
    "multiple",
    "muted",
    "nomodule",
    "novalidate",
    "open",
    "playsinline",
    "readonly",
    "required",
    "reversed",
    "selected"
  ]);
  function mount_component(component, target, anchor, customElement) {
    const { fragment, on_mount, on_destroy, after_update } = component.$$;
    fragment && fragment.m(target, anchor);
    if (!customElement) {
      add_render_callback(() => {
        const new_on_destroy = on_mount.map(run).filter(is_function);
        if (on_destroy) {
          on_destroy.push(...new_on_destroy);
        } else {
          run_all(new_on_destroy);
        }
        component.$$.on_mount = [];
      });
    }
    after_update.forEach(add_render_callback);
  }
  function destroy_component(component, detaching) {
    const $$ = component.$$;
    if ($$.fragment !== null) {
      run_all($$.on_destroy);
      $$.fragment && $$.fragment.d(detaching);
      $$.on_destroy = $$.fragment = null;
      $$.ctx = [];
    }
  }
  function make_dirty(component, i) {
    if (component.$$.dirty[0] === -1) {
      dirty_components.push(component);
      schedule_update();
      component.$$.dirty.fill(0);
    }
    component.$$.dirty[i / 31 | 0] |= 1 << i % 31;
  }
  function init(component, options, instance2, create_fragment2, not_equal, props, dirty = [-1]) {
    const parent_component = current_component;
    set_current_component(component);
    const $$ = component.$$ = {
      fragment: null,
      ctx: null,
      props,
      update: noop,
      not_equal,
      bound: blank_object(),
      on_mount: [],
      on_destroy: [],
      on_disconnect: [],
      before_update: [],
      after_update: [],
      context: new Map(parent_component ? parent_component.$$.context : options.context || []),
      callbacks: blank_object(),
      dirty,
      skip_bound: false
    };
    let ready = false;
    $$.ctx = instance2 ? instance2(component, options.props || {}, (i, ret, ...rest) => {
      const value = rest.length ? rest[0] : ret;
      if ($$.ctx && not_equal($$.ctx[i], $$.ctx[i] = value)) {
        if (!$$.skip_bound && $$.bound[i])
          $$.bound[i](value);
        if (ready)
          make_dirty(component, i);
      }
      return ret;
    }) : [];
    $$.update();
    ready = true;
    run_all($$.before_update);
    $$.fragment = create_fragment2 ? create_fragment2($$.ctx) : false;
    if (options.target) {
      if (options.hydrate) {
        start_hydrating();
        const nodes = children(options.target);
        $$.fragment && $$.fragment.l(nodes);
        nodes.forEach(detach);
      } else {
        $$.fragment && $$.fragment.c();
      }
      if (options.intro)
        transition_in(component.$$.fragment);
      mount_component(component, options.target, options.anchor, options.customElement);
      end_hydrating();
      flush();
    }
    set_current_component(parent_component);
  }
  var SvelteElement;
  if (typeof HTMLElement === "function") {
    SvelteElement = class extends HTMLElement {
      constructor() {
        super();
        this.attachShadow({ mode: "open" });
      }
      connectedCallback() {
        const { on_mount } = this.$$;
        this.$$.on_disconnect = on_mount.map(run).filter(is_function);
        for (const key in this.$$.slotted) {
          this.appendChild(this.$$.slotted[key]);
        }
      }
      attributeChangedCallback(attr2, _oldValue, newValue) {
        this[attr2] = newValue;
      }
      disconnectedCallback() {
        run_all(this.$$.on_disconnect);
      }
      $destroy() {
        destroy_component(this, 1);
        this.$destroy = noop;
      }
      $on(type, callback) {
        const callbacks = this.$$.callbacks[type] || (this.$$.callbacks[type] = []);
        callbacks.push(callback);
        return () => {
          const index = callbacks.indexOf(callback);
          if (index !== -1)
            callbacks.splice(index, 1);
        };
      }
      $set($$props) {
        if (this.$$set && !is_empty($$props)) {
          this.$$.skip_bound = true;
          this.$$set($$props);
          this.$$.skip_bound = false;
        }
      }
    };
  }
  var SvelteComponent = class {
    $destroy() {
      destroy_component(this, 1);
      this.$destroy = noop;
    }
    $on(type, callback) {
      const callbacks = this.$$.callbacks[type] || (this.$$.callbacks[type] = []);
      callbacks.push(callback);
      return () => {
        const index = callbacks.indexOf(callback);
        if (index !== -1)
          callbacks.splice(index, 1);
      };
    }
    $set($$props) {
      if (this.$$set && !is_empty($$props)) {
        this.$$.skip_bound = true;
        this.$$set($$props);
        this.$$.skip_bound = false;
      }
    }
  };

  // src/App.svelte
  var { window: window_1 } = globals;
  function get_each_context(ctx, list, i) {
    const child_ctx = ctx.slice();
    child_ctx[18] = list[i];
    return child_ctx;
  }
  function get_each_context_1(ctx, list, i) {
    const child_ctx = ctx.slice();
    child_ctx[21] = list[i];
    child_ctx[23] = i;
    return child_ctx;
  }
  function get_each_context_2(ctx, list, i) {
    const child_ctx = ctx.slice();
    child_ctx[24] = list[i];
    return child_ctx;
  }
  function create_each_block_2(ctx) {
    let div;
    let t_value = ctx[24][0] + "";
    let t;
    let div_style_value;
    return {
      c() {
        div = element("div");
        t = text(t_value);
        attr(div, "class", "cell svelte-oncm9j");
        attr(div, "style", div_style_value = `width: ${ctx[6]}px; height: ${ctx[7]}px; line-height: ${ctx[7]}px; opacity: ${ctx[24][1] * 100}%`);
      },
      m(target, anchor) {
        insert(target, div, anchor);
        append(div, t);
      },
      p(ctx2, dirty) {
        if (dirty & 16 && t_value !== (t_value = ctx2[24][0] + ""))
          set_data(t, t_value);
        if (dirty & 16 && div_style_value !== (div_style_value = `width: ${ctx2[6]}px; height: ${ctx2[7]}px; line-height: ${ctx2[7]}px; opacity: ${ctx2[24][1] * 100}%`)) {
          attr(div, "style", div_style_value);
        }
      },
      d(detaching) {
        if (detaching)
          detach(div);
      }
    };
  }
  function create_each_block_1(ctx) {
    let div;
    let t;
    let div_style_value;
    let each_value_2 = ctx[21];
    let each_blocks = [];
    for (let i = 0; i < each_value_2.length; i += 1) {
      each_blocks[i] = create_each_block_2(get_each_context_2(ctx, each_value_2, i));
    }
    return {
      c() {
        div = element("div");
        for (let i = 0; i < each_blocks.length; i += 1) {
          each_blocks[i].c();
        }
        t = space();
        attr(div, "class", "row svelte-oncm9j");
        attr(div, "style", div_style_value = `height: ${ctx[7]}px; ` + (ctx[23] % 2 === 1 ? `padding-left: ${ctx[6] / 2}px` : ""));
      },
      m(target, anchor) {
        insert(target, div, anchor);
        for (let i = 0; i < each_blocks.length; i += 1) {
          each_blocks[i].m(div, null);
        }
        append(div, t);
      },
      p(ctx2, dirty) {
        if (dirty & 208) {
          each_value_2 = ctx2[21];
          let i;
          for (i = 0; i < each_value_2.length; i += 1) {
            const child_ctx = get_each_context_2(ctx2, each_value_2, i);
            if (each_blocks[i]) {
              each_blocks[i].p(child_ctx, dirty);
            } else {
              each_blocks[i] = create_each_block_2(child_ctx);
              each_blocks[i].c();
              each_blocks[i].m(div, t);
            }
          }
          for (; i < each_blocks.length; i += 1) {
            each_blocks[i].d(1);
          }
          each_blocks.length = each_value_2.length;
        }
      },
      d(detaching) {
        if (detaching)
          detach(div);
        destroy_each(each_blocks, detaching);
      }
    };
  }
  function create_if_block_2(ctx) {
    let t0;
    let a;
    let t2;
    let mounted;
    let dispose;
    return {
      c() {
        t0 = text("You have died to death. ");
        a = element("a");
        a.textContent = "Restart";
        t2 = text(".");
        attr(a, "href", "#");
      },
      m(target, anchor) {
        insert(target, t0, anchor);
        insert(target, a, anchor);
        insert(target, t2, anchor);
        if (!mounted) {
          dispose = listen(a, "click", ctx[5]);
          mounted = true;
        }
      },
      p: noop,
      d(detaching) {
        if (detaching)
          detach(t0);
        if (detaching)
          detach(a);
        if (detaching)
          detach(t2);
        mounted = false;
        dispose();
      }
    };
  }
  function create_if_block_1(ctx) {
    let t0;
    let t1;
    return {
      c() {
        t0 = text(ctx[2]);
        t1 = text(" connected players.");
      },
      m(target, anchor) {
        insert(target, t0, anchor);
        insert(target, t1, anchor);
      },
      p(ctx2, dirty) {
        if (dirty & 4)
          set_data(t0, ctx2[2]);
      },
      d(detaching) {
        if (detaching)
          detach(t0);
        if (detaching)
          detach(t1);
      }
    };
  }
  function create_if_block(ctx) {
    let t0;
    let t1;
    let t2;
    return {
      c() {
        t0 = text("Your health is ");
        t1 = text(ctx[1]);
        t2 = text(".");
      },
      m(target, anchor) {
        insert(target, t0, anchor);
        insert(target, t1, anchor);
        insert(target, t2, anchor);
      },
      p(ctx2, dirty) {
        if (dirty & 2)
          set_data(t1, ctx2[1]);
      },
      d(detaching) {
        if (detaching)
          detach(t0);
        if (detaching)
          detach(t1);
        if (detaching)
          detach(t2);
      }
    };
  }
  function create_each_block(ctx) {
    let li;
    let t0_value = ctx[18][0] + "";
    let t0;
    let t1;
    let t2_value = ctx[18][2] + "";
    let t2;
    let t3;
    let t4_value = ctx[18][1] + "";
    let t4;
    return {
      c() {
        li = element("li");
        t0 = text(t0_value);
        t1 = text(" x");
        t2 = text(t2_value);
        t3 = text(": ");
        t4 = text(t4_value);
      },
      m(target, anchor) {
        insert(target, li, anchor);
        append(li, t0);
        append(li, t1);
        append(li, t2);
        append(li, t3);
        append(li, t4);
      },
      p(ctx2, dirty) {
        if (dirty & 8 && t0_value !== (t0_value = ctx2[18][0] + ""))
          set_data(t0, t0_value);
        if (dirty & 8 && t2_value !== (t2_value = ctx2[18][2] + ""))
          set_data(t2, t2_value);
        if (dirty & 8 && t4_value !== (t4_value = ctx2[18][1] + ""))
          set_data(t4, t4_value);
      },
      d(detaching) {
        if (detaching)
          detach(li);
      }
    };
  }
  function create_fragment(ctx) {
    let h1;
    let t1;
    let div2;
    let div0;
    let t2;
    let div1;
    let t3;
    let t4;
    let t5;
    let ul;
    let mounted;
    let dispose;
    let each_value_1 = ctx[4];
    let each_blocks_1 = [];
    for (let i = 0; i < each_value_1.length; i += 1) {
      each_blocks_1[i] = create_each_block_1(get_each_context_1(ctx, each_value_1, i));
    }
    let if_block0 = ctx[0] && create_if_block_2(ctx);
    let if_block1 = ctx[2] && create_if_block_1(ctx);
    let if_block2 = ctx[1] && create_if_block(ctx);
    let each_value = ctx[3];
    let each_blocks = [];
    for (let i = 0; i < each_value.length; i += 1) {
      each_blocks[i] = create_each_block(get_each_context(ctx, each_value, i));
    }
    return {
      c() {
        h1 = element("h1");
        h1.textContent = "EWO3 Memetic Edition";
        t1 = space();
        div2 = element("div");
        div0 = element("div");
        for (let i = 0; i < each_blocks_1.length; i += 1) {
          each_blocks_1[i].c();
        }
        t2 = space();
        div1 = element("div");
        if (if_block0)
          if_block0.c();
        t3 = space();
        if (if_block1)
          if_block1.c();
        t4 = space();
        if (if_block2)
          if_block2.c();
        t5 = space();
        ul = element("ul");
        for (let i = 0; i < each_blocks.length; i += 1) {
          each_blocks[i].c();
        }
        attr(div0, "class", "game-display svelte-oncm9j");
        attr(div1, "class", "controls");
        attr(div2, "class", "wrapper svelte-oncm9j");
      },
      m(target, anchor) {
        insert(target, h1, anchor);
        insert(target, t1, anchor);
        insert(target, div2, anchor);
        append(div2, div0);
        for (let i = 0; i < each_blocks_1.length; i += 1) {
          each_blocks_1[i].m(div0, null);
        }
        append(div2, t2);
        append(div2, div1);
        if (if_block0)
          if_block0.m(div1, null);
        append(div1, t3);
        if (if_block1)
          if_block1.m(div1, null);
        append(div1, t4);
        if (if_block2)
          if_block2.m(div1, null);
        append(div1, t5);
        append(div1, ul);
        for (let i = 0; i < each_blocks.length; i += 1) {
          each_blocks[i].m(ul, null);
        }
        if (!mounted) {
          dispose = [
            listen(window_1, "keydown", ctx[8]),
            listen(window_1, "keyup", ctx[9])
          ];
          mounted = true;
        }
      },
      p(ctx2, [dirty]) {
        if (dirty & 208) {
          each_value_1 = ctx2[4];
          let i;
          for (i = 0; i < each_value_1.length; i += 1) {
            const child_ctx = get_each_context_1(ctx2, each_value_1, i);
            if (each_blocks_1[i]) {
              each_blocks_1[i].p(child_ctx, dirty);
            } else {
              each_blocks_1[i] = create_each_block_1(child_ctx);
              each_blocks_1[i].c();
              each_blocks_1[i].m(div0, null);
            }
          }
          for (; i < each_blocks_1.length; i += 1) {
            each_blocks_1[i].d(1);
          }
          each_blocks_1.length = each_value_1.length;
        }
        if (ctx2[0]) {
          if (if_block0) {
            if_block0.p(ctx2, dirty);
          } else {
            if_block0 = create_if_block_2(ctx2);
            if_block0.c();
            if_block0.m(div1, t3);
          }
        } else if (if_block0) {
          if_block0.d(1);
          if_block0 = null;
        }
        if (ctx2[2]) {
          if (if_block1) {
            if_block1.p(ctx2, dirty);
          } else {
            if_block1 = create_if_block_1(ctx2);
            if_block1.c();
            if_block1.m(div1, t4);
          }
        } else if (if_block1) {
          if_block1.d(1);
          if_block1 = null;
        }
        if (ctx2[1]) {
          if (if_block2) {
            if_block2.p(ctx2, dirty);
          } else {
            if_block2 = create_if_block(ctx2);
            if_block2.c();
            if_block2.m(div1, t5);
          }
        } else if (if_block2) {
          if_block2.d(1);
          if_block2 = null;
        }
        if (dirty & 8) {
          each_value = ctx2[3];
          let i;
          for (i = 0; i < each_value.length; i += 1) {
            const child_ctx = get_each_context(ctx2, each_value, i);
            if (each_blocks[i]) {
              each_blocks[i].p(child_ctx, dirty);
            } else {
              each_blocks[i] = create_each_block(child_ctx);
              each_blocks[i].c();
              each_blocks[i].m(ul, null);
            }
          }
          for (; i < each_blocks.length; i += 1) {
            each_blocks[i].d(1);
          }
          each_blocks.length = each_value.length;
        }
      },
      i: noop,
      o: noop,
      d(detaching) {
        if (detaching)
          detach(h1);
        if (detaching)
          detach(t1);
        if (detaching)
          detach(div2);
        destroy_each(each_blocks_1, detaching);
        if (if_block0)
          if_block0.d();
        if (if_block1)
          if_block1.d();
        if (if_block2)
          if_block2.d();
        destroy_each(each_blocks, detaching);
        mounted = false;
        run_all(dispose);
      }
    };
  }
  var GRIDSIZE = 33;
  var SIZE = 16;
  function instance($$self, $$props, $$invalidate) {
    let dead = false;
    let health;
    let players;
    let inventory = [];
    let ws;
    const connect = () => {
      ws = new WebSocket(window.location.protocol === "https:" ? "wss://ewo.osmarks.net/" : "ws://localhost:8080/");
      ws.addEventListener("message", (ev) => {
        const data = JSON.parse(ev.data);
        if (data.Display) {
          const newGrid = blankGrid();
          for (const [q, r, c, o] of data.Display.nearby) {
            const col = q + (r - (r & 1)) / 2;
            const row = r;
            newGrid[row + OFFSET][col + OFFSET] = [c, o];
          }
          $$invalidate(4, grid = newGrid);
          $$invalidate(1, health = data.Display.health);
          $$invalidate(3, inventory = data.Display.inventory);
        }
        if (data === "Dead") {
          $$invalidate(0, dead = true);
        }
        if (data.PlayerCount) {
          $$invalidate(2, players = data.PlayerCount);
        }
        for (const key of keysDown) {
          const input = INPUTS[key];
          if (input) {
            ws.send(JSON.stringify(input));
          }
        }
        keysDown = new Set(Array.from(keysDown).map((k) => !keysCleared.has(k)));
        keysCleared = new Set();
      });
      ws.addEventListener("close", (ev) => {
        console.warn("oh no");
      });
    };
    const reconnect = () => {
      if (ws)
        ws.close();
      connect();
    };
    const restart = (ev) => {
      ev.preventDefault();
      $$invalidate(0, dead = false);
      reconnect();
    };
    const OFFSET = Math.floor(GRIDSIZE / 2);
    const HORIZ = Math.sqrt(3) * SIZE;
    const VERT = 3 / 2 * SIZE;
    const blankGrid = () => new Array(GRIDSIZE).fill(null).map(() => new Array(GRIDSIZE).fill("\u200B"));
    let grid = blankGrid();
    let keysDown = new Set();
    let keysCleared = new Set();
    const keydown = (ev) => {
      keysDown.add(ev.key);
    };
    const keyup = (ev) => {
      keysCleared.add(ev.key);
    };
    const INPUTS = {
      "w": "UpLeft",
      "e": "UpRight",
      "a": "Left",
      "d": "Right",
      "z": "DownLeft",
      "x": "DownRight",
      "f": "Dig"
    };
    connect();
    return [dead, health, players, inventory, grid, restart, HORIZ, VERT, keydown, keyup];
  }
  var App = class extends SvelteComponent {
    constructor(options) {
      super();
      init(this, options, instance, create_fragment, safe_not_equal, {});
    }
  };
  var App_default = App;

  // src/app.js
  new App_default({
    target: document.body
  });
})();
