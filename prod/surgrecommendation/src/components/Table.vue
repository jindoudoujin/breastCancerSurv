<template>
  <a-table id="table" :columns="columns" :data-source="data" :pagination="false">
    <a slot="action" slot-scope="text" href="javascript:;">action {{text.name}}</a>
  </a-table>
</template>
<script>
const columns = [
  { title: 'Surgery Type', width: 80, dataIndex: 'type', key: 'type'},
  { title: 'Hazard Rate', width: 80, dataIndex: 'rate', key: 'rate', 
    sorter: (a, b) => a.rate - b.rate,
    sortDirections: ['descend', 'ascend'],
  },
];
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
]
export default {
  props: {
    rate: Array
  },
  data() {
    return {
      columns,
    };
  },
  updated() {
    console.log(this.rate);
  },
  computed: {
    data() {
      let tmp = []
      this.rate.map((item, index) => tmp.push({
        type: surgeryArr[index],
        rate: (""+item).substring(0, 7)
      }))
      return tmp;
    }
  }
};
</script>

<style lang="less" scoped>
#table  {
  width: 800px;
}
</style>