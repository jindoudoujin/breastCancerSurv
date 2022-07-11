<template>
  <div id="main"></div>
</template>

<script>
import * as echarts from "echarts";
const surgeryArr = [
  "Bilateral mastectomy",
  "Extended radical mastectomy",
  "Local tumor destruction",
  "Mastectomy",
  "Modified radical mastectomy",
  "None",
  "Partial mastectomy",
  "Radical mastectomy",
  "Subcutaneous mastectomy",
  "Total (simple) mastectomy",
];
export default {
  name: "my-figure",
  props: {
    series: Array,
  },
  mounted() {
    console.log(this.series);
    this.data = this.series;
    const chartDom = document.getElementById("main");
    this.myChart = echarts.init(chartDom);
    this.option && this.myChart.setOption(this.option);
    this.series && this.run();
  },
  watch: {
    series(newValue) {
      this.data = newValue;
      this.myChart.clear();
      this.myChart.dispose();
      const chartDom = document.getElementById("main");
      this.myChart = echarts.init(chartDom);
      this.data && this.run();
    },
  },
  data() {
    return {
      option: {},
      myChart: null,
      data: [],
    };
  },
  methods: {
    run() {
      let min = 1;
      let datasetList = [];
      let seriesList = [];
      this.data.map((item, index) => {
        // dataset
        var datasetId = "dataset_" + index;
        let temp = [["Type", "Probability", "Month"]];
        item.forEach((val, idx) => {
          temp.push([surgeryArr[index], val, idx + 0]);
        });
        if (item[59] < min) {
          min = item[59];
        }
        datasetList.push({
          id: datasetId,
          source: temp,
        });
        // series
        seriesList.push({
          type: "line",
          datasetId: datasetId,
          showSymbol: false,
          name: surgeryArr[index],
          endLabel: {
            show: true,
            formatter: function (params) {
              return (
                params.value[0] +
                ": " +
                Math.floor(params.value[1] * 1000) / 1000
              );
            },
          },
          labelLayout: {
            moveOverlap: "shiftY",
          },
          emphasis: {
            focus: "series",
          },
          encode: {
            x: "Month",
            y: "Probability",
            label: ["Type", "Probability"],
            itemName: "Month",
            tooltip: ["Probability"],
          },
        });
      });
      this.option = {
        animationDuration: 5000,
        dataset: [...datasetList],
        title: {
          text: "Survial Probability of 10 surgeries in 60 months",
        },
        tooltip: {
          order: "valueDesc",
          trigger: "axis",
          // formatter: 'Month - {b}'
        },
        xAxis: {
          type: "category",
          nameLocation: "middle",
          name: "Month",
          nameGap: "20",
          axisTick: {
            // alignWithLabel: 1,
          },
        },
        yAxis: {
          name: "Survial Probability",
          min: Math.floor(min * 100) / 100,
          max: 1.0,
          show: true,
          axisLine: {
            show: true,
          },
        },
        grid: {
          right: 250,
          width: 850,
          bottom: 75,
        },
        series: seriesList,
        toolbox: {
          show: true,
          feature: {
            dataZoom: {
              show: true,
            },
            saveAsImage: { 
              show: true 
            },
          },
          left: "75%",
          top: 10,
        },
        dataZoom: [
          {
            type: "slider",
            show: true,
            xAxisIndex: [0],
            top: "93%",
            start: 0,
            end: 100,
          },
          {
            type: "slider",
            show: true,
            yAxisIndex: [0],
            left: "2%",
            start: 0,
            end: 100,
          },
          {
            type: "inside",
            xAxisIndex: [0],
            start: 0,
            end: 100,
          },
          {
            type: "inside",
            yAxisIndex: [0],
            start: 0,
            end: 100,
          },
        ],
      };
      this.myChart.setOption(this.option);
    },
  },
};
</script>

<style lang="less" scoped>
#main {
  height: 600px;
  width: 1200px;
}
</style>