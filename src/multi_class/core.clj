(ns multi-class.core
  (:require [clojure.core.matrix :as m :refer [mmul array ecount]]
            [clojure.core.matrix.operators :as op]))

(m/set-current-implementation :vectorz)

(defn- sigmoid [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn- softmax [xs]
  (let [sum-ex (reduce #(+ %1 (Math/exp %2)) 0 xs)]
    (map #(/ (Math/exp %) sum-ex) xs)))

(defn- has-one? [coll]
  (= 1 (ecount coll)))

(defn- rand-vec [n]
  (repeatedly n #(dec (rand 2))))

(defn- init-model [layers]
  (array (map (fn [[input-n node-n]]
                (array (repeatedly node-n #(array (rand-vec input-n)))))
           (partition 2 1 layers))))

(defn inference [input model]
  (loop [input (array input)
         layer model]
    (if (seq layer)
      (recur (let [output (mmul (first layer) input)]
               (if (has-one? layer)
                 (softmax output)
                 (map sigmoid output)))
        (rest layer))
      input)))
