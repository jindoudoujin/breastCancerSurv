<template>
  <div id="page">
    <a-form-model
      id="input-container"
      :model="form"
      :label-col="labelCol"
      :wrapper-col="wrapperCol"
    >
      <section class="input-section" id="Demographic">
        <h4>Demographic</h4>
        <!-- 年龄 -->
        <a-form-model-item required label="Age">
          <a-select v-model="form.age" placeholder="please select your age">
            <a-select-option value="1"> 15-19 years </a-select-option>
            <a-select-option value="2"> 20-24 years </a-select-option>
            <a-select-option value="3"> 25-29 years </a-select-option>
            <a-select-option value="4"> 30-34 years </a-select-option>
            <a-select-option value="5"> 35-39 years </a-select-option>
            <a-select-option value="6"> 40-44 years </a-select-option>
            <a-select-option value="7"> 45-49 years </a-select-option>
          </a-select>
        </a-form-model-item>
        <!-- 婚姻状况 -->
        <a-form-model-item required label="Marital status at diagnosis">
          <a-select
            v-model="form.marital"
            placeholder="please select your marital status"
          >
            <a-select-option value="1"> Divorced </a-select-option>
            <a-select-option value="2">
              Married (including common law)
            </a-select-option>
            <a-select-option value="3"> Separated </a-select-option>
            <a-select-option value="4">
              Single (never married)
            </a-select-option>
            <a-select-option value="5"> Unknown </a-select-option>
            <a-select-option value="6">
              Unmarried or Domestic Partner
            </a-select-option>
            <a-select-option value="7"> Widowed </a-select-option>
          </a-select>
        </a-form-model-item>
      </section>
      <!-- 受体类型 -->
      <section class="input-section" id="Receptors">
        <h4>Receptors</h4>
        <a-form-model-item required label="Breast Subtype">
          <a-select
            v-model="form.subtype"
            placeholder="please select your breast subtype"
          >
            <a-select-option value="1"> Luminal B </a-select-option>
            <a-select-option value="2"> Luminal A </a-select-option>
            <a-select-option value="3"> HER2 enriched </a-select-option>
            <a-select-option value="4"> Triple Negative </a-select-option>
          </a-select>
        </a-form-model-item>
      </section>
      <section class="input-section" id="Stage">
        <h4>Stage</h4>
        <a-form-model-item required
          :label-col="{ span: 5 }"
          :wrapper-col="{ span: 16 }"
          class="label-input short"
          label="T"
        >
          <a-tree-select
            v-model="form.T"
            style="width: 100%"
            :dropdown-style="{ maxHeight: '400px', overflow: 'auto' }"
            allow-clear
            tree-default-expand-all
          >
            <a-tree-select-node key="T0" value="1" title="T0" />
            <a-tree-select-node
              key="T1"
              :selectable="false"
              value="100"
              title="T1"
            >
              <a-tree-select-node key="T1a" value="2" title="T1a" />
              <a-tree-select-node key="T1b" value="3" title="T1b" />
              <a-tree-select-node key="T1c" value="4" title="T1c" />
              <a-tree-select-node key="T1mic" value="5" title="T1mic" />
            </a-tree-select-node>
            <a-tree-select-node key="T2" value="6" title="T2" />
            <a-tree-select-node key="T3" value="7" title="T3" />
            <a-tree-select-node
              key="T4"
              :selectable="false"
              value="40"
              title="T4"
            >
              <a-tree-select-node key="T4a" value="8" title="T4a" />
              <a-tree-select-node key="T4b" value="9" title="T4b" />
              <a-tree-select-node key="T4c" value="10" title="T4c" />
              <a-tree-select-node key="T4d" value="11" title="T4d" />
            </a-tree-select-node>
          </a-tree-select>
        </a-form-model-item>
        <a-form-model-item required
          :label-col="{ span: 13 }"
          :wrapper-col="{ span: 11 }"
          class="label-input"
          label="CS Tumor Size/Ext Eval"
        >
          <a-select
            v-model="form.CST"
            placeholder="please select your breast subtype"
          >
            <a-select-option value="1"> 0 </a-select-option>
            <a-select-option value="2"> 1 </a-select-option>
            <a-select-option value="3"> 3 </a-select-option>
            <a-select-option value="4"> 5 </a-select-option>
            <a-select-option value="5"> 6 </a-select-option>
          </a-select>
        </a-form-model-item>
        <a-form-model-item required
          :label-col="{ span: 5 }"
          :wrapper-col="{ span: 16 }"
          class="label-input short"
          label="N"
        >
          <a-select v-model="form.N">
            <a-select-option value="1"> N0 </a-select-option>
            <a-select-option value="2"> N1 </a-select-option>
            <a-select-option value="3"> N2 </a-select-option>
            <a-select-option value="4"> N3 </a-select-option>
          </a-select>
        </a-form-model-item>
        <a-form-model-item required
          :label-col="{ span: 13 }"
          :wrapper-col="{ span: 11 }"
          class="label-input"
          label="CS Reg Node Eval"
        >
          <a-select v-model="form.CSN">
            <a-select-option value="1"> 0 </a-select-option>
            <a-select-option value="2"> 1 </a-select-option>
            <a-select-option value="3"> 2 </a-select-option>
            <a-select-option value="4"> 3 </a-select-option>
            <a-select-option value="5"> 5 </a-select-option>
            <a-select-option value="6"> 6 </a-select-option>
          </a-select>
        </a-form-model-item>
        <a-form-model-item required
          :label-col="{ span: 5 }"
          :wrapper-col="{ span: 16 }"
          class="label-input short"
          label="M"
        >
          <a-select v-model="form.M">
            <a-select-option value="1"> M0 </a-select-option>
          </a-select>
        </a-form-model-item>
        <a-form-model-item required
          :label-col="{ span: 13 }"
          :wrapper-col="{ span: 11 }"
          class="label-input"
          label="CS Mets Eval"
        >
          <a-select v-model="form.CSM">
            <a-select-option value="1"> 0 </a-select-option>
            <a-select-option value="2"> 1 </a-select-option>
            <a-select-option value="3"> 3 </a-select-option>
            <a-select-option value="4"> 5 </a-select-option>
            <a-select-option value="5"> 6 </a-select-option>
          </a-select>
        </a-form-model-item>
      </section>
      <section class="input-section" id="Grade">
        <h4>Grade</h4>
        <a-form-model-item required label="Grade">
          <a-select
            v-model="form.grade"
            placeholder="please select your breast subtype"
          >
            <a-select-option value="1">
              Well differentiated; Grade I
            </a-select-option>
            <a-select-option value="2">
              Moderately differentiated; Grade II
            </a-select-option>
            <a-select-option value="3">
              Poorly differentiated; Grade III
            </a-select-option>
            <a-select-option value="4">
              Undifferentiated; anaplastic; Grade IV
            </a-select-option>
          </a-select>
        </a-form-model-item>
      </section>
      <section class="input-section" id="Type">
        <h4>Type</h4>
        <a-form-model-item required label="Histologic Type ICD-O-3">
          <a-select
            v-model="form.histologic"
            placeholder="please select your breast subtype"
          >
            <a-select-option value="1">
              Infiltrating duct carcinoma
            </a-select-option>
            <a-select-option value="2"> Comedocarcinoma </a-select-option>
            <a-select-option value="3">
              Secretory carcinoma of breast
            </a-select-option>
            <a-select-option value="4">
              Intraductal papillary adenocarcinoma with invasion
            </a-select-option>
            <a-select-option value="5"> Intracystic carcinoma </a-select-option>
            <a-select-option value="6">
              Ductal carcinoma, micropapillary
            </a-select-option>
            <a-select-option value="7"> Lobular carcinoma </a-select-option>
            <a-select-option value="8">
              Infiltrating ductular carcinoma
            </a-select-option>
            <a-select-option value="9">
              Infiltrating duct and lobular carcinoma
            </a-select-option>
            <a-select-option value="10">
              Infiltrating duct mixed with other types of carcinoma
            </a-select-option>
            <a-select-option value="11">
              Infiltrating lobular mixed with other types of carcinoma
            </a-select-option>
            <a-select-option value="12">
              Polymorphous low grade adenocarcinoma
            </a-select-option>
          </a-select>
        </a-form-model-item>
        <a-form-model-item required label="Laterality ">
          <a-select
            v-model="form.laterality"
            placeholder="please select your breast subtype"
          >
            <a-select-option value="1">
              Bilateral, single primary
            </a-select-option>
            <a-select-option value="2">
              Left - origin of primary
            </a-select-option>
            <a-select-option value="3">
              Only one side - side unspecified
            </a-select-option>
            <a-select-option value="4">
              Paired site, but no information concerning laterality
            </a-select-option>
            <a-select-option value="5">
              Right - origin of primary
            </a-select-option>
          </a-select>
        </a-form-model-item>
      </section>
      <section class="input-section" id="Treatment">
        <h4>Treatment</h4>
        <a-form-model-item required label="Radiation recode">
          <a-select v-model="form.radiation_recode">
            <a-select-option value="1"> Beam radiation </a-select-option>
            <a-select-option value="2">
              Combination of beam with implants or isotopes
            </a-select-option>
            <a-select-option value="3"> None/Unknown </a-select-option>
            <a-select-option value="4">
              Radiation, NOS method or source not specified
            </a-select-option>
            <a-select-option value="5">
              Radioactive implants (includes brachytherapy)
            </a-select-option>
            <a-select-option value="6"> Radioisotopes </a-select-option>
            <a-select-option value="7">
              Recommended, unknown if administered
            </a-select-option>
            <a-select-option value="8"> Refused </a-select-option>
          </a-select>
        </a-form-model-item>
        <a-form-model-item required label="Chemotherapy recode">
          <a-select v-model="form.chemotherapy_recode">
            <a-select-option value="1"> No/Unknown </a-select-option>
            <a-select-option value="2"> Yes </a-select-option>
          </a-select>
        </a-form-model-item>
      </section>
      <a-form-model-item required :wrapper-col="{ span: 14, offset: 8 }">
        <a-button type="primary" :disabled="spinning || !active" @click="onSubmit">
          Submit
        </a-button>
        <a-button style="margin-left: 30px" @click="onReset"> Reset </a-button>
      </a-form-model-item>
    </a-form-model>
    <div id="figure-container">
      <a-spin size="large" :spinning="spinning">
        <a-tabs default-active-key="1">
          <a-tab-pane key="1" tab="hazard rate table">
            <Table :rate="rate" />
          </a-tab-pane>
          <a-tab-pane key="2" tab="survival curve">
            <Figure :series="series" />
          </a-tab-pane>
        </a-tabs>
      </a-spin>
    </div>
  </div>
</template>

<script>
import Table from "./components/Table.vue";
import Figure from "./components/Figure.vue";
export default {
  name: "Page",
  components: {
    Table,
    Figure,
  },
  async mounted() {
    console.log(this.$http);
  },
  data() {
    return {
      spinning: false,
      labelCol: { span: 10 },
      wrapperCol: { span: 14 },
      form: {
        age: "",
        marital: "",
        subtype: "",
        T: "",
        CST: "",
        N: [],
        CSN: [],
        M: "",
        CSM: "",
        grade: "",
        histologic: "",
        laterality: "",
        radiation_recode: "",
        chemotherapy_recode: "",
      },
      rate: [],
      series: [],
    };
  },
  computed: {
    active() {
      for(let _ in this.form) {
        if (!this.form[_]) {
          return false
        }
      }
      return true
    }
  },
  methods: {
    async onSubmit() {
      this.rate = [];
      this.series = [];
      this.spinning = true;
      try {
        const res = await this.$http.post("/breastcancer/hello", JSON.stringify(this.form), {
          headers: { "Content-Type": "application/json" },
        });
        // let {rate, series} = res.data;
        this.spinning = false;
        this.rate = res.data.rate;
        this.series = res.data.series;
      } catch (error) {
        this.$notification.error({
          message: "Error",
          description: "网络异常",
          duration: 3
        });
        this.spinning = false;
      }
    },
    onReset() {
      for(let _ in this.form) {
        this.form[_] = ''
      }
    },
  },
};
</script>

<style lang="less">
html {
  padding: 30px;
  min-width: 1850px;
}

#input-container {
  width: 550px;
}

.input-section {
  border: 1px solid #ebedf0;
  padding: 20px;
  border-radius: 2px;
  position: relative;
  margin-bottom: 20px;

  h4 {
    position: absolute;
    top: -10px;
    background-color: #ffffff;
    padding: 0 5px;
    font-weight: bold;
  }

  &#Stage {
    display: flex;
    flex-wrap: wrap;

    .label-input {
      width: 64%;

      &.short {
        width: 36%;
      }
    }
  }
}

#figure-container {
  margin-left: 40px;
  width: 1200px;
}

#page {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  display: flex;
}
</style>
